use wasm_bindgen::prelude::*;
use ndarray::{Array1, Array2, Array3};
mod main;
use main::{Transformer, Tokenizer, Sampler};

// Re-export necessary structs with wasm_bindgen
#[wasm_bindgen]
#[derive(Debug, Copy, Clone)]
pub struct WasmConfig {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

#[wasm_bindgen]
impl WasmConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        vocab_size: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        }
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "(text: string) => void")]
    pub type TokenCallback;

    #[wasm_bindgen(method, structural, js_name = "call")]
    fn call(this: &TokenCallback, this_arg: &JsValue, text: &str);
}


#[wasm_bindgen]
pub struct WasmTransformer {
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
}

#[wasm_bindgen]
impl WasmTransformer {
    #[wasm_bindgen(constructor)]
    pub fn new(model_bytes: &[u8], tokenizer_bytes: &[u8], temperature: f32, topp: f32) -> Result<WasmTransformer, JsValue> {
        console_error_panic_hook::set_once();
        
        let transformer = Transformer::from_bytes(model_bytes)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
            
        let tokenizer = Tokenizer::from_bytes(
            tokenizer_bytes, 
            transformer.config.vocab_size
        ).map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let sampler = Sampler::new(
            transformer.config.vocab_size,
            temperature,
            topp,
            42  // Fixed seed for now
        );
        
        Ok(Self {
            transformer,
            tokenizer,
            sampler,
        })
    }


    #[wasm_bindgen]
    pub fn forward(&mut self, token: i32, pos: i32) {
        self.transformer.forward(token, pos);
    }
    
    #[wasm_bindgen]
    pub fn get_next_token(&mut self, pos: usize, prompt_tokens: Option<Vec<usize>>) -> usize {
        if let Some(tokens) = prompt_tokens {
            if pos < tokens.len() - 1 {
                return tokens[pos + 1];
            }
        }
        self.sampler.sample(self.transformer.state.logits.as_slice_mut().unwrap())
    }

    #[wasm_bindgen]
    pub fn encode(&mut self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text, true, false)
    }

    #[wasm_bindgen]
    pub fn decode(&self, prev_token: usize, token: usize) -> String {
        self.tokenizer.decode(prev_token, token).to_string()
    }
}

// Called when the wasm module is instantiated
#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}