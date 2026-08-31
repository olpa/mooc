use candle_core::{Device, Tensor};
use ch1_tfmech::attention;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct Fixture {
    q: Vec<Vec<Vec<f32>>>,
    k: Vec<Vec<Vec<f32>>>,
    v: Vec<Vec<Vec<f32>>>,
    out: Vec<Vec<Vec<f32>>>,
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

fn tensor3(data: &[Vec<Vec<f32>>]) -> Tensor {
    let d0 = data.len();
    let d1 = data[0].len();
    let d2 = data[0][0].len();
    let flat: Vec<f32> = data.iter().flatten().flatten().copied().collect();
    Tensor::from_vec(flat, (d0, d1, d2), &Device::Cpu).expect("failed to build tensor")
}

fn assert_close(actual: &Tensor, expected: &[Vec<Vec<f32>>], tol: f32) {
    let actual = actual.to_vec3::<f32>().expect("expected a 3d f32 tensor");
    assert_eq!(actual.len(), expected.len(), "batch dimension mismatch");
    for (a_batch, e_batch) in actual.iter().zip(expected) {
        assert_eq!(a_batch.len(), e_batch.len(), "seq_len dimension mismatch");
        for (a_row, e_row) in a_batch.iter().zip(e_batch) {
            assert_eq!(a_row.len(), e_row.len(), "value dimension mismatch");
            for (a, e) in a_row.iter().zip(e_row) {
                assert!(
                    (a - e).abs() < tol,
                    "value mismatch: actual={a} expected={e} (tol={tol})"
                );
            }
        }
    }
}

fn run_case(fixture_name: &str) {
    let fixture = load_fixture(fixture_name);
    let q = tensor3(&fixture.q);
    let k = tensor3(&fixture.k);
    let v = tensor3(&fixture.v);

    let out = attention(&q, &k, &v).expect("attention failed");

    assert_close(&out, &fixture.out, 1e-4);
}

#[test]
fn attention_small() {
    run_case("attention_small");
}

#[test]
fn attention_small_diff_v() {
    run_case("attention_small_diff_v");
}

#[test]
fn attention_fixed() {
    run_case("attention_fixed");
}
