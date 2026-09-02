use candle_core::{DType, Device, Tensor};
use ch2_genloop::generate;
use serde::Deserialize;
use std::fs;
use std::path::Path;

// The fixtures stand in a real model with a deterministic per-token lookup
// table (row i = logits produced when the last input token is i), so the
// fixture exercises the generation loop itself (argmax, concat, EOS stop)
// rather than a model implementation. See python-reference/gen_generate.py.
#[derive(Deserialize)]
struct Fixture {
    table: Vec<Vec<f32>>,
    prompt_ids: Vec<Vec<u32>>,
    max_tokens: usize,
    eos_token_id: u32,
    out: Vec<Vec<u32>>,
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

fn tensor2_u32(data: &[Vec<u32>]) -> Tensor {
    let d0 = data.len();
    let d1 = data[0].len();
    let flat: Vec<u32> = data.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (d0, d1), &Device::Cpu).expect("failed to build tensor")
}

fn run_case(fixture_name: &str) {
    let fixture = load_fixture(fixture_name);
    let vocab_size = fixture.table.len();
    let table: Vec<f32> = fixture.table.into_iter().flatten().collect();
    let table = Tensor::from_vec(table, (vocab_size, vocab_size), &Device::Cpu)
        .expect("failed to build table tensor");

    // model(input_ids: (1, seq_len) u32) -> logits: (1, seq_len, vocab_size)
    let model = |input_ids: &Tensor| -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let flat = input_ids.flatten_all()?.to_dtype(DType::U32)?;
        table.index_select(&flat, 0)?.reshape((1, seq_len, vocab_size))
    };

    let prompt_ids = tensor2_u32(&fixture.prompt_ids);

    let out = generate(model, &prompt_ids, fixture.max_tokens, fixture.eos_token_id)
        .expect("generate failed");

    let out = out.to_vec2::<u32>().expect("expected a 2d u32 tensor");
    assert_eq!(out, fixture.out);
}

#[test]
fn generate_runs_to_max_tokens() {
    run_case("generate_runs_to_max_tokens");
}

#[test]
fn generate_small_vocab() {
    run_case("generate_small_vocab");
}

#[test]
fn generate_stops_at_eos() {
    run_case("generate_stops_at_eos");
}
