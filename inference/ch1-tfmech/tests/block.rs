use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use ch1_tfmech::TransformerBlock;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct Fixture {
    num_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    x: Vec<Vec<Vec<f32>>>,
    norm1_weight: Vec<f32>,
    q_proj_weight: Vec<Vec<f32>>,
    q_proj_bias: Vec<f32>,
    k_proj_weight: Vec<Vec<f32>>,
    k_proj_bias: Vec<f32>,
    v_proj_weight: Vec<Vec<f32>>,
    v_proj_bias: Vec<f32>,
    out_proj_weight: Vec<Vec<f32>>,
    out_proj_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    w1_weight: Vec<Vec<f32>>,
    w3_weight: Vec<Vec<f32>>,
    w2_weight: Vec<Vec<f32>>,
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

fn tensor2(data: &[Vec<f32>]) -> Tensor {
    let d0 = data.len();
    let d1 = data[0].len();
    let flat: Vec<f32> = data.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (d0, d1), &Device::Cpu).expect("failed to build tensor")
}

fn tensor1(data: &[f32]) -> Tensor {
    Tensor::from_vec(data.to_vec(), data.len(), &Device::Cpu).expect("failed to build tensor")
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
    let hidden_dim = fixture.num_heads * fixture.head_dim;

    let mut tensors = HashMap::new();
    tensors.insert("norm1.weight".to_string(), tensor1(&fixture.norm1_weight));
    tensors.insert(
        "attn.q_proj.weight".to_string(),
        tensor2(&fixture.q_proj_weight),
    );
    tensors.insert(
        "attn.q_proj.bias".to_string(),
        tensor1(&fixture.q_proj_bias),
    );
    tensors.insert(
        "attn.k_proj.weight".to_string(),
        tensor2(&fixture.k_proj_weight),
    );
    tensors.insert(
        "attn.k_proj.bias".to_string(),
        tensor1(&fixture.k_proj_bias),
    );
    tensors.insert(
        "attn.v_proj.weight".to_string(),
        tensor2(&fixture.v_proj_weight),
    );
    tensors.insert(
        "attn.v_proj.bias".to_string(),
        tensor1(&fixture.v_proj_bias),
    );
    tensors.insert(
        "attn.out_proj.weight".to_string(),
        tensor2(&fixture.out_proj_weight),
    );
    tensors.insert(
        "attn.out_proj.bias".to_string(),
        tensor1(&fixture.out_proj_bias),
    );
    tensors.insert("norm2.weight".to_string(), tensor1(&fixture.norm2_weight));
    tensors.insert("ffn.w1.weight".to_string(), tensor2(&fixture.w1_weight));
    tensors.insert("ffn.w3.weight".to_string(), tensor2(&fixture.w3_weight));
    tensors.insert("ffn.w2.weight".to_string(), tensor2(&fixture.w2_weight));

    let mut vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let block = TransformerBlock::new(
        hidden_dim,
        fixture.num_heads,
        fixture.intermediate_dim,
        &mut vb,
    )
    .expect("failed to build TransformerBlock");

    let x = tensor3(&fixture.x);
    let out = block.forward(&x).expect("forward failed");

    assert_close(&out, &fixture.out, 1e-4);
}

#[test]
fn block_small() {
    run_case("block_small");
}
