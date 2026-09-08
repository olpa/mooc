use candle_core::{Device, Tensor};
use ch2_genloop::decode_attention;
use serde::Deserialize;
use std::fs;
use std::path::Path;

// decode_attention takes plain tensors (no model): q_new/k_new/v_new are a
// single new position, k_cache/v_cache hold the previously cached positions.
// See python-reference/gen_decode_attention.py.
#[derive(Deserialize)]
struct Fixture {
    q_new: Vec<Vec<Vec<f32>>>,
    k_cache: Vec<Vec<Vec<f32>>>,
    v_cache: Vec<Vec<Vec<f32>>>,
    k_new: Vec<Vec<Vec<f32>>>,
    v_new: Vec<Vec<Vec<f32>>>,
    out: Vec<Vec<Vec<f32>>>,
    k_cache_out: Vec<Vec<Vec<f32>>>,
    v_cache_out: Vec<Vec<Vec<f32>>>,
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

fn tensor3_f32(data: &[Vec<Vec<f32>>]) -> Tensor {
    let d0 = data.len();
    let d1 = data[0].len();
    let d2 = data[0][0].len();
    let flat: Vec<f32> = data.iter().flatten().flatten().copied().collect();
    Tensor::from_vec(flat, (d0, d1, d2), &Device::Cpu).expect("failed to build tensor")
}

fn run_case(fixture_name: &str) {
    let fixture = load_fixture(fixture_name);

    let q_new = tensor3_f32(&fixture.q_new);
    let k_cache = tensor3_f32(&fixture.k_cache);
    let v_cache = tensor3_f32(&fixture.v_cache);
    let k_new = tensor3_f32(&fixture.k_new);
    let v_new = tensor3_f32(&fixture.v_new);

    let (out, k_cache_out, v_cache_out) = decode_attention(&q_new, &k_cache, &v_cache, &k_new, &v_new)
        .expect("decode_attention failed");

    let out = out.to_vec3::<f32>().expect("expected a 3d f32 tensor");
    let k_cache_out = k_cache_out
        .to_vec3::<f32>()
        .expect("expected a 3d f32 tensor");
    let v_cache_out = v_cache_out
        .to_vec3::<f32>()
        .expect("expected a 3d f32 tensor");

    assert_close3(&out, &fixture.out);
    // k_cache_out/v_cache_out are plain concatenation, no arithmetic, so they
    // must match exactly.
    assert_eq!(k_cache_out, fixture.k_cache_out);
    assert_eq!(v_cache_out, fixture.v_cache_out);
}

// matmul/softmax accumulate in a different order than PyTorch, so `out`
// only matches up to floating-point rounding.
fn assert_close3(actual: &[Vec<Vec<f32>>], expected: &[Vec<Vec<f32>>]) {
    const EPS: f32 = 1e-5;
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.len(), e.len());
        for (a, e) in a.iter().zip(e) {
            assert_eq!(a.len(), e.len());
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
fn decode_attention_small() {
    run_case("decode_attention_small");
}
