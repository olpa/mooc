use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use ch1_tfmech::Transformer;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Deserialize)]
struct LayerFixture {
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
}

#[derive(Deserialize)]
struct Fixture {
    vocab_size: usize,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    input_ids: Vec<Vec<u32>>,
    embed_weight: Vec<Vec<f32>>,
    layers: Vec<LayerFixture>,
    norm_weight: Vec<f32>,
    lm_head_weight: Vec<Vec<f32>>,
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

fn tensor2_u32(data: &[Vec<u32>]) -> Tensor {
    let d0 = data.len();
    let d1 = data[0].len();
    let flat: Vec<u32> = data.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (d0, d1), &Device::Cpu).expect("failed to build tensor")
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
    tensors.insert("embed.weight".to_string(), tensor2(&fixture.embed_weight));
    for (i, layer) in fixture.layers.iter().enumerate() {
        tensors.insert(
            format!("layers.{i}.norm1.weight"),
            tensor1(&layer.norm1_weight),
        );
        tensors.insert(
            format!("layers.{i}.attn.q_proj.weight"),
            tensor2(&layer.q_proj_weight),
        );
        tensors.insert(
            format!("layers.{i}.attn.q_proj.bias"),
            tensor1(&layer.q_proj_bias),
        );
        tensors.insert(
            format!("layers.{i}.attn.k_proj.weight"),
            tensor2(&layer.k_proj_weight),
        );
        tensors.insert(
            format!("layers.{i}.attn.k_proj.bias"),
            tensor1(&layer.k_proj_bias),
        );
        tensors.insert(
            format!("layers.{i}.attn.v_proj.weight"),
            tensor2(&layer.v_proj_weight),
        );
        tensors.insert(
            format!("layers.{i}.attn.v_proj.bias"),
            tensor1(&layer.v_proj_bias),
        );
        tensors.insert(
            format!("layers.{i}.attn.out_proj.weight"),
            tensor2(&layer.out_proj_weight),
        );
        tensors.insert(
            format!("layers.{i}.attn.out_proj.bias"),
            tensor1(&layer.out_proj_bias),
        );
        tensors.insert(
            format!("layers.{i}.norm2.weight"),
            tensor1(&layer.norm2_weight),
        );
        tensors.insert(
            format!("layers.{i}.ffn.w1.weight"),
            tensor2(&layer.w1_weight),
        );
        tensors.insert(
            format!("layers.{i}.ffn.w3.weight"),
            tensor2(&layer.w3_weight),
        );
        tensors.insert(
            format!("layers.{i}.ffn.w2.weight"),
            tensor2(&layer.w2_weight),
        );
    }
    tensors.insert("norm.weight".to_string(), tensor1(&fixture.norm_weight));
    tensors.insert(
        "lm_head.weight".to_string(),
        tensor2(&fixture.lm_head_weight),
    );

    let mut vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let model = Transformer::new(
        fixture.vocab_size,
        hidden_dim,
        fixture.num_layers,
        fixture.num_heads,
        fixture.intermediate_dim,
        &mut vb,
    )
    .expect("failed to build Transformer");

    let input_ids = tensor2_u32(&fixture.input_ids);
    let out = model.forward(&input_ids).expect("forward failed");

    assert_close(&out, &fixture.out, 1e-3);
}

#[test]
fn transformer_small() {
    run_case("transformer_small");
}
