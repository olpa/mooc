use candle_core::{DType, Device, Tensor};
use ch2_genloop::KVCache;
use serde::Deserialize;
use std::fs;
use std::path::Path;

// Exercises KVCache.update/get/advance directly (no model, no attention):
// a prefill-like multi-token write for two layers, followed by a
// decode-like single-token write, checkpointing `get` before/after each
// `advance` call. See python-reference/gen_kvcache.py.
//
// k/v passed to `update` are (n, num_heads, head_dim) -- no batch dim. This
// cache doesn't model batching at all (its own storage has no batch axis),
// so there's nothing to drop or add back on either `update` or `get`.
//
#[derive(Deserialize)]
struct Config {
    num_layers: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
}

#[derive(Deserialize)]
struct GetCheckpoint {
    seq_len: usize,
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct LayerGet {
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
}

#[derive(Deserialize)]
struct AfterAdvance1 {
    seq_len: usize,
    layer0: LayerGet,
    layer1: LayerGet,
}

#[derive(Deserialize)]
struct Fixture {
    config: Config,
    k0_a: Vec<Vec<Vec<f32>>>,
    v0_a: Vec<Vec<Vec<f32>>>,
    k1_a: Vec<Vec<Vec<f32>>>,
    v1_a: Vec<Vec<Vec<f32>>>,
    k0_b: Vec<Vec<Vec<f32>>>,
    v0_b: Vec<Vec<Vec<f32>>>,
    get_layer0_before_advance1: GetCheckpoint,
    get_after_advance1: AfterAdvance1,
    get_layer0_before_advance2: GetCheckpoint,
    get_layer0_after_advance2: GetCheckpoint,
}

fn load_fixture(name: &str) -> Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("python-reference/fixtures")
        .join(format!("{name}.json"));
    let data = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "failed to read fixture {path:?}: {e}. \
             Run `make fixtures` in python-reference/ first."
        )
    });
    serde_json::from_str(&data).expect("invalid fixture JSON")
}

// data: (n, num_heads, head_dim), n possibly 0 -> Tensor (n, num_heads, head_dim)
fn tensor3_f32(data: &[Vec<Vec<f32>>], num_heads: usize, head_dim: usize) -> Tensor {
    let d0 = data.len();
    let flat: Vec<f32> = data.iter().flatten().flatten().copied().collect();
    Tensor::from_vec(flat, (d0, num_heads, head_dim), &Device::Cpu).expect("failed to build tensor")
}

fn assert_close3(actual: &Tensor, expected: &[Vec<Vec<f32>>], num_heads: usize, head_dim: usize) {
    const EPS: f32 = 1e-5;
    let expected_t = tensor3_f32(expected, num_heads, head_dim);
    assert_eq!(actual.dims(), expected_t.dims());
    if expected.is_empty() {
        return;
    }
    let actual = actual.to_vec3::<f32>().expect("expected a 3d f32 tensor");
    for (a, e) in actual.iter().zip(expected) {
        for (a, e) in a.iter().zip(e) {
            for (a, e) in a.iter().zip(e) {
                assert!(
                    (a - e).abs() <= EPS,
                    "values differ beyond tolerance: {a} vs {e}"
                );
            }
        }
    }
}

#[test]
fn kvcache_update_get_advance() {
    let fixture = load_fixture("kvcache");
    let num_heads = fixture.config.num_heads;
    let head_dim = fixture.config.head_dim;

    let mut cache = KVCache::new(
        fixture.config.num_layers,
        fixture.config.max_seq_len,
        num_heads,
        head_dim,
        DType::F32,
        &Device::Cpu,
    )
    .expect("KVCache::new failed");

    let k0_a = tensor3_f32(&fixture.k0_a, num_heads, head_dim);
    let v0_a = tensor3_f32(&fixture.v0_a, num_heads, head_dim);
    let k1_a = tensor3_f32(&fixture.k1_a, num_heads, head_dim);
    let v1_a = tensor3_f32(&fixture.v1_a, num_heads, head_dim);
    let k0_b = tensor3_f32(&fixture.k0_b, num_heads, head_dim);
    let v0_b = tensor3_f32(&fixture.v0_b, num_heads, head_dim);

    cache.update(0, &k0_a, &v0_a).expect("update(0, a) failed");
    cache.update(1, &k1_a, &v1_a).expect("update(1, a) failed");

    // Before any advance(), get() must not see either write yet.
    assert_eq!(
        cache.get_seq_len(),
        fixture.get_layer0_before_advance1.seq_len
    );
    let (k, v) = cache.get(0).expect("get(0) failed");
    assert_close3(
        &k,
        &fixture.get_layer0_before_advance1.k,
        num_heads,
        head_dim,
    );
    assert_close3(
        &v,
        &fixture.get_layer0_before_advance1.v,
        num_heads,
        head_dim,
    );

    cache.advance(3);

    assert_eq!(cache.get_seq_len(), fixture.get_after_advance1.seq_len);
    let (k0, v0) = cache.get(0).expect("get(0) failed");
    assert_close3(
        &k0,
        &fixture.get_after_advance1.layer0.k,
        num_heads,
        head_dim,
    );
    assert_close3(
        &v0,
        &fixture.get_after_advance1.layer0.v,
        num_heads,
        head_dim,
    );
    let (k1, v1) = cache.get(1).expect("get(1) failed");
    assert_close3(
        &k1,
        &fixture.get_after_advance1.layer1.k,
        num_heads,
        head_dim,
    );
    assert_close3(
        &v1,
        &fixture.get_after_advance1.layer1.v,
        num_heads,
        head_dim,
    );

    cache.update(0, &k0_b, &v0_b).expect("update(0, b) failed");

    // The layer-0 write above must not be visible until advance() runs --
    // same contents as right after advance1.
    assert_eq!(
        cache.get_seq_len(),
        fixture.get_layer0_before_advance2.seq_len
    );
    let (k, v) = cache.get(0).expect("get(0) failed");
    assert_close3(
        &k,
        &fixture.get_layer0_before_advance2.k,
        num_heads,
        head_dim,
    );
    assert_close3(
        &v,
        &fixture.get_layer0_before_advance2.v,
        num_heads,
        head_dim,
    );

    cache.advance(1);

    assert_eq!(
        cache.get_seq_len(),
        fixture.get_layer0_after_advance2.seq_len
    );
    let (k, v) = cache.get(0).expect("get(0) failed");
    assert_close3(
        &k,
        &fixture.get_layer0_after_advance2.k,
        num_heads,
        head_dim,
    );
    assert_close3(
        &v,
        &fixture.get_layer0_after_advance2.v,
        num_heads,
        head_dim,
    );
}
