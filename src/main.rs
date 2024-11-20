// Keep these imports
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, BufReader, Write};
use memmap2::MmapOptions;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::slice;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayViewMut1};
use std::cmp::Ordering;
use std::time::Instant;
use std::io::Cursor;

// Define the macro before any usage
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => {
        web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(&format!($($t)*)))
    };
}


// =======================================================
// =================TRANSFORMER CONFIG====================
// =======================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}
#[derive(Debug, Clone)]
struct TransformerWeights {
    token_embedding_table: Array2<f32>,  // [vocab_size, dim]
    rms_att_weight: Array2<f32>,         // [n_layers, dim]
    wq: Array3<f32>,                     // [n_layers, dim, n_heads * head_size]
    wk: Array3<f32>,                     // [n_layers, dim, n_kv_heads * head_size]
    wv: Array3<f32>,                     // [n_layers, dim, n_kv_heads * head_size]
    wo: Array3<f32>,                     // [n_layers, n_heads * head_size, dim]
    rms_ffn_weight: Array2<f32>,         // [n_layers, dim]
    w1: Array3<f32>,                     // [n_layers, dim, hidden_dim]
    w2: Array3<f32>,                     // [n_layers, hidden_dim, dim]
    w3: Array3<f32>,                     // [n_layers, dim, hidden_dim]
    rms_final_weight: Array1<f32>,       // [dim]
    wcls: Array2<f32>,                   // [vocab_size, dim]
}

// =======================================================
// =====================TRANSFORMER=======================
// =======================================================
#[derive(Debug, Clone)]
pub struct RunState {
    x: Array1<f32>,
    xb: Array1<f32>,
    xb2: Array1<f32>,
    hb: Array1<f32>,
    hb2: Array1<f32>,
    q: Array1<f32>,
    k: Array1<f32>,
    v: Array1<f32>,
    att: Array2<f32>,
    pub logits: Array1<f32>,
    key_cache: Array3<f32>,
    value_cache: Array3<f32>,
}
impl RunState {
    fn new(config: &Config) -> Self {
        let kv_dim = (config.dim as usize * config.n_kv_heads as usize) / config.n_heads as usize;
        
        Self {
            x: Array1::zeros(config.dim),
            xb: Array1::zeros(config.dim),
            xb2: Array1::zeros(config.dim),
            hb: Array1::zeros(config.hidden_dim),
            hb2: Array1::zeros(config.hidden_dim),
            q: Array1::zeros(config.dim),
            k: Array1::zeros(kv_dim),
            v: Array1::zeros(kv_dim),
            att: Array2::zeros((config.n_heads, config.seq_len)),
            logits: Array1::zeros(config.vocab_size),
            key_cache: Array3::zeros((config.n_layers, config.seq_len, kv_dim)),
            value_cache: Array3::zeros((config.n_layers, config.seq_len, kv_dim)),
        }
    }
}

#[derive(Debug, Clone)]
#[repr(C)]
pub struct Transformer{
    pub config: Config,
    weights: TransformerWeights,
    pub state: RunState,
}

// ========== UTIL FUNCTIONS =========

fn load_checkpoint(checkpoint_path: &str) -> io::Result<(Config, TransformerWeights)> {
    let mut file = File::open(checkpoint_path)?;

    let mut buffer = [0i32; 7]; // 7 fields
    unsafe {
        let slice = slice::from_raw_parts_mut(
            buffer.as_mut_ptr() as *mut u8,
            buffer.len() * std::mem::size_of::<i32>()
        );
        file.read_exact(slice)?;
    }
    let config = Config {
        dim: buffer[0] as usize,
        hidden_dim: buffer[1] as usize,
        n_layers: buffer[2] as usize,
        n_heads: buffer[3] as usize,
        n_kv_heads: buffer[4] as usize,
        vocab_size: buffer[5] as usize,
        seq_len: buffer[6] as usize,
    };

    let shared_weights = config.vocab_size > 0;

    file.seek(SeekFrom::Start(0))?;

    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let weights_ptr = unsafe { mmap.as_ptr().add((7 * std::mem::size_of::<i32>())) as *const f32 };

    let head_size = (config.dim / config.n_heads) as usize;
    let mut offset = 0;


    // let mut arrays = Vec::new();
    // let sizes = [
    //     (config.vocab_size, config.dim),
    //     (config.n_layers, config.dim),
    //     (config.n_layers, config.dim, config.n_heads * head_size),
    //     (config.n_layers, config.dim, config.n_kv_heads * head_size),
    //     (config.n_layers, config.dim, config.n_kv_heads * head_size),
    //     (config.n_layers, config.n_heads * head_size, config.dim),
    //     (config.n_layers, config.dim),
    //     (config.n_layers, config.dim, config.hidden_dim),
    //     (config.n_layers, config.hidden_dim, config.dim),
    //     (config.n_layers, config.dim, config.hidden_dim),
    //     (config.dim,)
    // ];

    // unsafe {
    //     let mut curr_ptr = weights_ptr;
    //     for &size in &sizes {
    //         let total_elements = size.iter().product();
    //         let slice = std::slice::from_raw_parts(curr_ptr, total_elements);
    //         arrays.push(slice.to_vec());
    //         curr_ptr = curr_ptr.add(total_elements);
    //     }
    // }
    // Ok((Config, Self {
    //     token_embedding: Array2::from_shape_vec((sizes[0].0, sizes[0].1), arrays[0].clone())?,
    //     rms_att: Array2::from_shape_vec((sizes[1].0, sizes[1].1), arrays[1].clone())?,
    //     wq: Array3::from_shape_vec((sizes[2].0, sizes[2].1, sizes[2].2), arrays[2].clone())?,
    //     wk: Array3::from_shape_vec((sizes[3].0, sizes[3].1, sizes[3].2), arrays[3].clone())?,
    //     wv: Array3::from_shape_vec((sizes[4].0, sizes[4].1, sizes[4].2), arrays[4].clone())?,
    //     wo: Array3::from_shape_vec((sizes[5].0, sizes[5].1, sizes[5].2), arrays[5].clone())?,
    //     rms_ffn: Array2::from_shape_vec((sizes[6].0, sizes[6].1), arrays[6].clone())?,
    //     w1: Array3::from_shape_vec((sizes[7].0, sizes[7].1, sizes[7].2), arrays[7].clone())?,
    //     w2: Array3::from_shape_vec((sizes[8].0, sizes[8].1, sizes[8].2), arrays[8].clone())?,
    //     w3: Array3::from_shape_vec((sizes[9].0, sizes[9].1, sizes[9].2), arrays[9].clone())?,
    //     rms_final: Array1::from_shape_vec(sizes[10].0, arrays[10].clone())?,
    //     wcls: Array2::from_shape_vec((sizes[0].0, sizes[0].1), arrays[0].clone())?,
    // }))

    unsafe {
        let token_embedding_table = weights_ptr.add(offset);        
        offset += config.vocab_size * config.dim;

        let rms_att_weight = weights_ptr.add(offset); 
        offset += config.n_layers * config.dim;

        let wq = weights_ptr.add(offset);     
        offset += config.n_layers * config.dim * (config.n_heads * head_size);

        let wk = weights_ptr.add(offset);
        offset += config.n_layers * config.dim * (config.n_kv_heads * head_size);

        let wv = weights_ptr.add(offset);
        offset += config.n_layers * config.dim * (config.n_kv_heads * head_size);

        let wo = weights_ptr.add(offset);
        offset += config.n_layers * (config.n_heads * head_size) * config.dim;

        let rms_ffn_weight = weights_ptr.add(offset);
        offset += config.n_layers * config.dim;

        let w1 = weights_ptr.add(offset);
        offset += config.n_layers * config.dim * config.hidden_dim;

        let w2 = weights_ptr.add(offset);
        offset += config.n_layers * config.hidden_dim * config.dim;

        let w3 = weights_ptr.add(offset);
        offset += config.n_layers * config.dim * config.hidden_dim;

        let rms_final_weight = weights_ptr.add(offset);
        offset += config.vocab_size * config.dim;
        offset += config.dim;
        
        offset += config.seq_len * head_size; // Skip rope frequencies

        let wcls_ptr = if shared_weights { 
            token_embedding_table 
        } else { 
            weights_ptr.add(offset) 
        };

        // Now convert to ndarray types by copying the data
        let weights = TransformerWeights {
            token_embedding_table: Array2::from_shape_vec(
                (config.vocab_size, config.dim),
                std::slice::from_raw_parts(token_embedding_table, config.vocab_size * config.dim).to_vec()
            ).unwrap(),
            
            rms_att_weight: Array2::from_shape_vec(
                (config.n_layers, config.dim),
                std::slice::from_raw_parts(rms_att_weight, config.n_layers * config.dim).to_vec()
            ).unwrap(),
            
            wq: Array3::from_shape_vec(
                (config.n_layers, config.dim, config.n_heads * head_size),
                std::slice::from_raw_parts(wq, config.n_layers * config.dim * (config.n_heads * head_size)).to_vec()
            ).unwrap(),
            
            wk: Array3::from_shape_vec(
                (config.n_layers, config.dim, config.n_kv_heads * head_size),
                std::slice::from_raw_parts(wk, config.n_layers * config.dim * (config.n_kv_heads * head_size)).to_vec()
            ).unwrap(),
            
            wv: Array3::from_shape_vec(
                (config.n_layers, config.dim, config.n_kv_heads * head_size),
                std::slice::from_raw_parts(wv, config.n_layers * config.dim * (config.n_kv_heads * head_size)).to_vec()
            ).unwrap(),
            
            wo: Array3::from_shape_vec(
                (config.n_layers, config.n_heads * head_size, config.dim),
                std::slice::from_raw_parts(wo, config.n_layers * (config.n_heads * head_size) * config.dim).to_vec()
            ).unwrap(),
            
            rms_ffn_weight: Array2::from_shape_vec(
                (config.n_layers, config.dim),
                std::slice::from_raw_parts(rms_ffn_weight, config.n_layers * config.dim).to_vec()
            ).unwrap(),
            
            w1: Array3::from_shape_vec(
                (config.n_layers, config.hidden_dim, config.dim),
                std::slice::from_raw_parts(w1, config.n_layers * config.dim * config.hidden_dim).to_vec()
            ).unwrap(),
            
            w2: Array3::from_shape_vec(
                (config.n_layers, config.dim, config.hidden_dim),
                std::slice::from_raw_parts(w2, config.n_layers * config.hidden_dim * config.dim).to_vec()
            ).unwrap(),
            
            w3: Array3::from_shape_vec(
                (config.n_layers, config.hidden_dim, config.dim),
                std::slice::from_raw_parts(w3, config.n_layers * config.dim * config.hidden_dim).to_vec()
            ).unwrap(),
            
            rms_final_weight: Array1::from_shape_vec(
                config.dim,
                std::slice::from_raw_parts(rms_final_weight, config.dim).to_vec()
            ).unwrap(),
            
            wcls: if shared_weights {
                Array2::from_shape_vec(
                    (config.vocab_size, config.dim),
                    std::slice::from_raw_parts(token_embedding_table, config.vocab_size * config.dim).to_vec()
                ).unwrap()
            } else {
                Array2::from_shape_vec(
                    (config.vocab_size, config.dim),
                    std::slice::from_raw_parts(wcls_ptr, config.vocab_size * config.dim).to_vec()
                ).unwrap()
            },
        };

        Ok((config, weights))
    }
}


// OLD RMS NORM BELOW
fn rms_norm(out: &mut Array1<f32>, x: &Array1<f32>, weight: &ArrayView1<f32>) {
    // Calculate sum of squares
    let ss: f32 = x.iter().map(|&v| v * v).sum::<f32>();
    let norm_factor = 1.0 / (ss / x.len() as f32 + 1e-5).sqrt();
    
    // Normalize and scale with weights in-place
    out.zip_mut_with(x, |o, &x| {
        *o = x * norm_factor;
    });
    out.zip_mut_with(weight, |o, &w| {
        *o *= w;
    });
}

fn softmax(x: &mut ArrayViewMut1<f32>) {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    x.mapv_inplace(|a| (a - max).exp());
    let sum: f32 = x.sum();
    x.mapv_inplace(|a| a / sum);
}

impl Transformer {
    pub fn new(checkpoint_path: &str) -> io::Result<Self> {
        let (config, weights) = load_checkpoint(checkpoint_path)?;
        let state = RunState::new(&config);
    
        Ok(Self {
            config,
            weights,
            state,
        })
    }

    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        use std::io::{Error, ErrorKind};
        
        let mut cursor = Cursor::new(bytes);
        
        let mut buffer = [0i32; 7];
        unsafe {
            let slice = slice::from_raw_parts_mut(
                buffer.as_mut_ptr() as *mut u8,
                buffer.len() * std::mem::size_of::<i32>()
            );
            cursor.read_exact(slice)?;
        }
        
        for (i, &value) in buffer.iter().enumerate() {
            if value <= 0 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid config value at position {}: {}", i, value)
                ));
            }
        }
        
        let config = Config {
            dim: buffer[0] as usize,
            hidden_dim: buffer[1] as usize,
            n_layers: buffer[2] as usize,
            n_heads: buffer[3] as usize,
            n_kv_heads: buffer[4] as usize,
            vocab_size: buffer[5] as usize,
            seq_len: buffer[6] as usize,
        };

        let shared_weights = config.vocab_size > 0;
        cursor.set_position(0);

        let head_size = config.dim / config.n_heads;
        let mut weights_buffer = vec![0f32; bytes.len() / std::mem::size_of::<f32>()];
        
        unsafe {
            let slice = slice::from_raw_parts_mut(
                weights_buffer.as_mut_ptr() as *mut u8,
                weights_buffer.len() * std::mem::size_of::<f32>()
            );
            cursor.read_exact(slice)?;
        }
        
        let mut offset = 7; // Skip config integers

        let token_embedding_table = Array2::from_shape_vec(
            (config.vocab_size, config.dim),
            weights_buffer[offset..offset + config.vocab_size * config.dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.vocab_size * config.dim;
        
        let rms_att_weight = Array2::from_shape_vec(
            (config.n_layers, config.dim),
            weights_buffer[offset..offset + config.n_layers * config.dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim;
        
        let wq = Array3::from_shape_vec(
            (config.n_layers, config.dim, config.n_heads * head_size),
            weights_buffer[offset..offset + config.n_layers * config.dim * (config.n_heads * head_size)].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim * (config.n_heads * head_size);
        
        let wk = Array3::from_shape_vec(
            (config.n_layers, config.dim, config.n_kv_heads * head_size),
            weights_buffer[offset..offset + config.n_layers * config.dim * (config.n_kv_heads * head_size)].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim * (config.n_kv_heads * head_size);
        
        let wv = Array3::from_shape_vec(
            (config.n_layers, config.dim, config.n_kv_heads * head_size),
            weights_buffer[offset..offset + config.n_layers * config.dim * (config.n_kv_heads * head_size)].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim * (config.n_kv_heads * head_size);
        
        let wo = Array3::from_shape_vec(
            (config.n_layers, config.n_heads * head_size, config.dim),
            weights_buffer[offset..offset + config.n_layers * (config.n_heads * head_size) * config.dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * (config.n_heads * head_size) * config.dim;
        
        let rms_ffn_weight = Array2::from_shape_vec(
            (config.n_layers, config.dim),
            weights_buffer[offset..offset + config.n_layers * config.dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim;
        
        let w1 = Array3::from_shape_vec(
            (config.n_layers, config.hidden_dim, config.dim),
            weights_buffer[offset..offset + config.n_layers * config.dim * config.hidden_dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim * config.hidden_dim;
        
        let w2 = Array3::from_shape_vec(
            (config.n_layers,  config.dim, config.hidden_dim,),
            weights_buffer[offset..offset + config.n_layers * config.hidden_dim * config.dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.hidden_dim * config.dim;
        
        let w3 = Array3::from_shape_vec(
            (config.n_layers, config.hidden_dim, config.dim),
            weights_buffer[offset..offset + config.n_layers * config.dim * config.hidden_dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.n_layers * config.dim * config.hidden_dim;
        
        let rms_final_weight = Array1::from_shape_vec(
            config.dim,
            weights_buffer[offset..offset + config.dim].to_vec()
        ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
        offset += config.dim;
        
        let wcls = if shared_weights {
            token_embedding_table.clone()
        } else {
            Array2::from_shape_vec(
                (config.vocab_size, config.dim),
                weights_buffer[offset..offset + config.vocab_size * config.dim].to_vec()
            ).map_err(|e| Error::new(ErrorKind::InvalidData, e))?
        };
        
        let weights = TransformerWeights {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        };
        
        let state = RunState::new(&config);
        
        Ok(Self {
            config,
            weights,
            state,
        })
    }

    pub fn forward(&mut self, token: i32, pos: i32) {
        
        let dim = self.config.dim;
        let kv_dim = (self.config.dim * self.config.n_kv_heads) / self.config.n_heads;
        let kv_mul = self.config.n_heads / self.config.n_kv_heads;
        let head_size = dim / self.config.n_heads;

        // Copy token embedding
        let token_idx = token as usize;
        self.state.x.assign(&self.weights.token_embedding_table.slice(s![token_idx, ..]));

        // Process each layer
        for l in 0..self.config.n_layers {
            // let mut xb_temp = Array1::zeros(dim);
            
            // Attention rmsnorm
            rms_norm(
                &mut self.state.xb,
                &self.state.x,
                &self.weights.rms_att_weight.slice(s![l, ..]).into()
            );
        
            self.state.q.assign(&self.weights.wq.slice(s![l, .., ..]).dot(&self.state.xb));
            {
                let mut key_slice = self.state.key_cache.slice_mut(s![l, pos as usize, ..]);
                let mut value_slice = self.state.value_cache.slice_mut(s![l, pos as usize, ..]);

                key_slice.assign(&self.weights.wk.slice(s![l, .., ..]).dot(&self.state.xb));
                value_slice.assign(&self.weights.wv.slice(s![l, .., ..]).dot(&self.state.xb));
            }

            // RoPE rotations
            for i in (0..dim).step_by(2) {
                let head_dim = i % head_size;
                let freq = 1.0 / (10000.0f32.powf(head_dim as f32 / head_size as f32));
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                let rotn = if i < kv_dim { 2 } else { 1 };
                
                match rotn {
                    2 => {
                        // Apply to both Q and K
                        let (v0, v1) = (self.state.q[i], self.state.q[i + 1]);
                        self.state.q[i] = v0 * fcr - v1 * fci;
                        self.state.q[i + 1] = v0 * fci + v1 * fcr;
                 
                        let mut key_slice = self.state.key_cache.slice_mut(s![l, pos as usize, ..]);
                        let (v0, v1) = (key_slice[i], key_slice[i + 1]);
                        key_slice[i] = v0 * fcr - v1 * fci;
                        key_slice[i + 1] = v0 * fci + v1 * fcr;
                  
                    },
                    _ => {
                        // Apply only to Q
                        let (v0, v1) = (self.state.q[i], self.state.q[i + 1]);         
                  
                        self.state.q[i] = v0 * fcr - v1 * fci;
                        self.state.q[i + 1] = v0 * fci + v1 * fcr;
                    }
                }
            }

            for h in 0..self.config.n_heads {


                let mut att_array = Array1::zeros(pos as usize + 1);

                {
                    let q = self.state.q.slice(s![h * head_size..(h + 1) * head_size]);
                    for t in 0..=pos as usize {
                        let k_slice = self.state.key_cache.slice(s![l, t, (h / kv_mul) * head_size..((h / kv_mul) + 1) * head_size]);
                        let score = (q.to_owned() * k_slice).sum() / (head_size as f32).sqrt();
    
                        att_array[t] = score;
                    }
                }

                softmax(&mut att_array.view_mut());

                let mut head_output = Array1::zeros(head_size);
                for t in 0..=pos as usize {
                    let v_slice = self.state.value_cache.slice(s![l, t, (h / kv_mul) * head_size..((h / kv_mul) + 1) * head_size]);

                    head_output += &(&v_slice * att_array[t]);
                }

                self.state.xb.slice_mut(s![h * head_size..(h + 1) * head_size])
                    .assign(&head_output);
            }

            self.state.xb2.assign(&self.weights.wo.slice(s![l, .., ..]).dot(&self.state.xb));
            self.state.x += &self.state.xb2;

            // FFN
            rms_norm(
                &mut self.state.xb,
                &self.state.x,
                &self.weights.rms_ffn_weight.slice(s![l, ..]).into()
            );
            self.state.hb.assign(&self.weights.w1.slice(s![l, .., ..]).dot(&self.state.xb));
            self.state.hb2.assign(&self.weights.w3.slice(s![l, .., ..]).dot(&self.state.xb)); 
               
            self.state.hb.mapv_inplace(|x| x * (1.0 / (1.0 + (-x).exp())));
            self.state.hb *= &self.state.hb2;
    
            self.state.xb.assign(&self.weights.w2.slice(s![l, .., ..]).dot(&self.state.hb));
    
            self.state.x += &self.state.xb;
        }

        {
            let mut x_temp = Array1::zeros(dim);
            rms_norm(
                &mut x_temp,
                &self.state.x,
                &self.weights.rms_final_weight.view()
            );
            self.state.x.assign(&x_temp);
        }

        self.state.logits.assign(&self.state.x.dot(&self.weights.wcls.t()));
    }
}

// =======================================================
// ======================TOKENIZER========================
// =======================================================

#[derive(Debug)]
struct TokenIndex {
    str_val: String,
    id: usize,
}

impl TokenIndex {
    fn new(str_val: String, id: usize) -> Self {
        Self { str_val, id }
    }
}
#[derive(Debug)]
pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    max_token_length: usize,
    byte_pieces: [u8; 512],  // Fixed-size array instead of Vec
    sorted_vocab: Option<Vec<TokenIndex>>,
}
impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: usize) -> io::Result<Self>{
        let mut byte_pieces = [0u8; 512];
        for i in 0..256 {
            byte_pieces[i * 2] = i as u8;
            byte_pieces[i * 2 + 1] = 0;
        }

        let file = File::open(tokenizer_path)?;
        let mut reader = BufReader::new(file);

        let mut max_token_length_bytes = [0u8; 4];
        reader.read_exact(&mut max_token_length_bytes)?;
        let max_token_length = i32::from_ne_bytes(max_token_length_bytes) as usize;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        for _ in 0..vocab_size{

            let mut score_bytes = [0u8; 4];
            reader.read_exact(&mut score_bytes)?;
            let score = f32::from_ne_bytes(score_bytes);
            vocab_scores.push(score);


            let mut len_bytes = [0u8; 4];
            reader.read_exact(&mut len_bytes)?;
            let len = i32::from_ne_bytes(len_bytes) as usize;


            let mut string_data = vec![0u8; len];
            reader.read_exact(&mut string_data)?;
            
            // Convert to string, handle invalid UTF-8 by replacing invalid sequences
            let string = String::from_utf8_lossy(&string_data).into_owned();
            vocab.push(string);
        }

        Ok( Self {
            vocab, 
            vocab_scores, 
            max_token_length, 
            byte_pieces,
            sorted_vocab: None,
        })
    }

    pub fn from_bytes(bytes: &[u8], vocab_size: usize) -> io::Result<Self> {
        let mut cursor = Cursor::new(bytes);
        let mut byte_pieces = [0u8; 512];
        
        for i in 0..256 {
            byte_pieces[i * 2] = i as u8;
            byte_pieces[i * 2 + 1] = 0;
        }
        
        let mut max_token_length_bytes = [0u8; 4];
        cursor.read_exact(&mut max_token_length_bytes)?;
        let max_token_length = i32::from_ne_bytes(max_token_length_bytes) as usize;
        
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);
        
        for _ in 0..vocab_size {
            let mut score_bytes = [0u8; 4];
            cursor.read_exact(&mut score_bytes)?;
            let score = f32::from_ne_bytes(score_bytes);
            vocab_scores.push(score);
            
            let mut len_bytes = [0u8; 4];
            cursor.read_exact(&mut len_bytes)?;
            let len = i32::from_ne_bytes(len_bytes) as usize;
            
            let mut string_data = vec![0u8; len];
            cursor.read_exact(&mut string_data)?;
            
            let string = String::from_utf8_lossy(&string_data).into_owned();
            vocab.push(string);
        }
        
        Ok(Self {
            vocab,
            vocab_scores,
            max_token_length,
            byte_pieces,
            sorted_vocab: None,
        })
    }

    pub fn encode(&mut self, text: &str, bos: bool, eos: bool) -> Vec<usize>{
        let mut tokens = Vec::with_capacity(text.len() + 2);

        if self.sorted_vocab.is_none() {
            let mut sorted = Vec::with_capacity(self.vocab_scores.len());
            for (i, str_val) in self.vocab.iter().enumerate() {
                sorted.push(TokenIndex::new(str_val.clone(), i));
            }
            sorted.sort_by(|a, b| a.str_val.cmp(&b.str_val));
            self.sorted_vocab = Some(sorted);
        }

        if bos {
            tokens.push(1);
        }

        if !text.is_empty() {
            if let Some(dummy_id) = self.str_lookup(" ") {
                tokens.push(dummy_id);
            }
        }

        let mut str_buffer = String::with_capacity(self.max_token_length * 2 + 3);
        
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            str_buffer.clear();
            str_buffer.push(c);
            
            // Check for UTF-8 continuation bytes using char boundaries
            while let Some(&next_c) = chars.peek() {
                if str_buffer.len() >= 4 {
                    break;
                }
                str_buffer.push(next_c);
                chars.next();
            }

            // Look up the token
            if let Some(id) = self.str_lookup(&str_buffer) {
                tokens.push(id);
            } else {
                // Fallback: encode each byte separately
                for byte in str_buffer.bytes() {
                    tokens.push((byte as usize) + 3);
                }
            }
        }

        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = None;
            let mut best_idx = None;

            // Find the best pair to merge
            for i in 0..tokens.len().saturating_sub(1) {
                let merged = format!("{}{}", 
                    &self.vocab[tokens[i]], 
                    &self.vocab[tokens[i + 1]]
                );
                
                if let Some(id) = self.str_lookup(&merged) {
                    let score = self.vocab_scores[id];
                    if score > best_score {
                        best_score = score;
                        best_id = Some(id);
                        best_idx = Some(i);
                    }
                }
            }

            // If no pairs can be merged, we're done
            match (best_id, best_idx) {
                (Some(id), Some(idx)) => {
                    tokens[idx] = id;
                    tokens.remove(idx + 1);
                }
                _ => break,
            }
        }

        if eos {
            tokens.push(2);
        }

        tokens
    }

    pub fn decode<'a>(&'a self, prev_token: usize, token: usize) -> &'a str {
        let piece = &self.vocab[token];
        
        if prev_token == 1 && piece.starts_with(' ') {
            return &piece[1..];
        }

        if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
            if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
                let idx = byte_val as usize * 2;
                return unsafe { std::str::from_utf8_unchecked(&self.byte_pieces[idx..idx + 1]) };
            }
        }

        piece
    }

    fn str_lookup(&self, s: &str) -> Option<usize> {
        self.sorted_vocab.as_ref().and_then(|vocab| {
            vocab.binary_search_by(|item| item.str_val.as_str().cmp(s))
                .ok()
                .map(|idx| vocab[idx].id)
        })
    }
}



// =======================================================
// =======================SAMPLER=========================
// =======================================================

#[derive(Debug, Clone, Copy)]
struct ProbIndex {
    prob: f32,
    index: usize,
}

impl PartialEq for ProbIndex {
    fn eq(&self, other: &Self) -> bool {
        self.prob == other.prob
    }
}

impl PartialOrd for ProbIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for descending sort
        other.prob.partial_cmp(&self.prob)
    }
}

#[derive(Debug)]
pub struct Sampler {
    vocab_size: usize,
    probindex: Vec<ProbIndex>,
    temperature: f32,
    topp: f32,
    rng: ChaCha8Rng, // Using ChaCha8Rng for good performance and determinism
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, seed: u64) -> Self {
        Self {
            vocab_size,
            probindex: vec![ProbIndex { prob: 0.0, index: 0 }; vocab_size],
            temperature,
            topp,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    fn sample_argmax(probabilities: &[f32]) -> usize {
        probabilities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .unwrap_or(0)
    }

    fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
        let mut cdf = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cdf += prob;
            if coin < cdf {
                return i;
            }
        }
        probabilities.len() - 1 // Handle rounding errors
    }

    fn sample_topp(&mut self, probabilities: &[f32], topp: f32, coin: f32) -> usize {
        let cutoff = (1.0 - topp) / (probabilities.len() - 1) as f32;
        
        // Fill probindex with values above cutoff
        let mut n0 = 0;
        for (i, &prob) in probabilities.iter().enumerate() {
            if prob >= cutoff {
                self.probindex[n0] = ProbIndex { prob, index: i };
                n0 += 1;
            }
        }

        // Sort in descending order of probabilities
        self.probindex[..n0].sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(Ordering::Equal));

        // Find truncation point where cumulative probability exceeds topp
        let mut cumulative_prob = 0.0;
        let mut last_idx = n0 - 1;
        for (i, prob_idx) in self.probindex[..n0].iter().enumerate() {
            cumulative_prob += prob_idx.prob;
            if cumulative_prob > topp {
                last_idx = i;
                break;
            }
        }

        // Sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0.0;
        for prob_idx in self.probindex[..=last_idx].iter() {
            cdf += prob_idx.prob;
            if r < cdf {
                return prob_idx.index;
            }
        }

        self.probindex[last_idx].index // Handle rounding errors
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> usize {
        assert_eq!(logits.len(), self.vocab_size, "Logits length must match vocab size");

        if self.temperature == 0.0 {
            // Greedy argmax sampling
            return Self::sample_argmax(logits);
        }

        // Apply temperature
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }

        // Apply softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        
        for logit in logits.iter_mut() {
            *logit = (*logit - max_logit).exp();
            sum += *logit;
        }
        
        for logit in logits.iter_mut() {
            *logit /= sum;
        }

        // Generate random number using rand crate
        let coin: f32 = self.rng.gen();

        if self.topp <= 0.0 || self.topp >= 1.0 {
            Self::sample_mult(logits, coin)
        } else {
            self.sample_topp(logits, self.topp, coin)
        }
    }
}


fn generate(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<&str>,
    steps: usize,
) -> io::Result<()> {
    let prompt = prompt.unwrap_or("");
    let prompt_tokens = tokenizer.encode(prompt, true, false);

    if prompt_tokens.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "Expected at least 1 prompt token"));
    }

    let mut start = None;
    let mut token = prompt_tokens[0];
    let mut pos = 0;

    while pos < steps {
        transformer.forward(token as i32, pos as i32);
        
        let next = if pos < prompt_tokens.len() - 1 {
            prompt_tokens[pos + 1]
        } else {
            sampler.sample(transformer.state.logits.as_slice_mut().expect("Logits not initialized"))
        };
        pos += 1;

        if next == 1 { break; }

        print!("{}", tokenizer.decode(token, next));
        io::stdout().flush()?;
        token = next;

        if start.is_none() {
            start = Some(Instant::now());
        }
    }
    println!();

    if pos > 1 {
        if let Some(start_time) = start {
            let duration = start_time.elapsed();
            eprintln!(
                "achieved tok/s: {}",
                (pos - 1) as f64 / duration.as_secs_f64()
            );
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <checkpoint> [options]", args[0]);
        std::process::exit(1);
    }

    let checkpoint_path = &args[1];
    let mut temperature = 1.0f32;
    let mut topp = 0.9f32;
    let mut steps = 256;
    let mut prompt = None;
    let mut seed = 0u64;
    let mut tokenizer_path = "tokenizer.bin";

    let mut i = 2;
    while i < args.len() {
        if i + 1 >= args.len() || !args[i].starts_with('-') {
            eprintln!("Invalid argument format");
            std::process::exit(1);
        }

        match args[i].as_str() {
            "-t" => temperature = args[i + 1].parse().unwrap_or(0.0),
            "-p" => topp = args[i + 1].parse().unwrap_or(0.5),
            "-s" => seed = args[i + 1].parse().unwrap_or(43),
            "-n" => steps = args[i + 1].parse().unwrap_or(256),
            "-i" => prompt = Some(args[i + 1].as_str()),
            "-z" => tokenizer_path = &args[i + 1],
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 2;
    }

    if seed == 0 {
        seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
    temperature = temperature.max(0.0);
    topp = topp.clamp(0.0, 1.0);
    
    let mut transformer = Transformer::new(checkpoint_path)?;
    if steps == 0 || steps > transformer.config.seq_len {
        steps = transformer.config.seq_len;
    }

    let mut tokenizer = Tokenizer::new(tokenizer_path, transformer.config.vocab_size)?;
    let mut sampler = Sampler::new(transformer.config.vocab_size, temperature, topp, seed);

    generate(&mut transformer, &mut tokenizer, &mut sampler, prompt, steps)?;

    Ok(())
}