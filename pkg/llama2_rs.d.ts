/* tslint:disable */
/* eslint-disable */
export function main(): void;
export class WasmConfig {
  free(): void;
  /**
   * @param {number} dim
   * @param {number} hidden_dim
   * @param {number} n_layers
   * @param {number} n_heads
   * @param {number} n_kv_heads
   * @param {number} vocab_size
   * @param {number} seq_len
   */
  constructor(dim: number, hidden_dim: number, n_layers: number, n_heads: number, n_kv_heads: number, vocab_size: number, seq_len: number);
}
export class WasmTransformer {
  free(): void;
  /**
   * @param {Uint8Array} model_bytes
   * @param {Uint8Array} tokenizer_bytes
   * @param {number} temperature
   * @param {number} topp
   */
  constructor(model_bytes: Uint8Array, tokenizer_bytes: Uint8Array, temperature: number, topp: number);
  /**
   * @param {string} prompt
   * @param {number} max_tokens
   * @returns {string}
   */
  generate(prompt: string, max_tokens: number): string;
  /**
   * @param {number} token
   * @param {number} pos
   */
  forward(token: number, pos: number): void;
  /**
   * @param {number} prev_token
   * @param {number} pos
   * @param {Uint32Array | undefined} [prompt_tokens]
   * @returns {number}
   */
  get_next_token(prev_token: number, pos: number, prompt_tokens?: Uint32Array): number;
  /**
   * @param {string} text
   * @returns {Uint32Array}
   */
  encode(text: string): Uint32Array;
  /**
   * @param {number} prev_token
   * @param {number} token
   * @returns {string}
   */
  decode(prev_token: number, token: number): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_wasmconfig_free: (a: number, b: number) => void;
  readonly wasmconfig_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly __wbg_wasmtransformer_free: (a: number, b: number) => void;
  readonly wasmtransformer_new: (a: number, b: number, c: number, d: number, e: number, f: number) => Array;
  readonly wasmtransformer_generate: (a: number, b: number, c: number, d: number) => Array;
  readonly wasmtransformer_forward: (a: number, b: number, c: number) => void;
  readonly wasmtransformer_get_next_token: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly wasmtransformer_encode: (a: number, b: number, c: number) => Array;
  readonly wasmtransformer_decode: (a: number, b: number, c: number) => Array;
  readonly main: () => void;
  readonly __wbindgen_export_0: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
