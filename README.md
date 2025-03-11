# Llama2.rs

A Rust implementation of [llama2.c](https://github.com/karpathy/llama2.c), complete with a [WASM-backed GitHub Page](https://qharo.github.io/llama2.rs/) running the 15M param model in the browser. Find more information [here](https://qharo.github.io/projects/llama2.rs/).

### Setup

1. To run this script, all you have to do after you clone it

```sh
git clone https://github.com/qharo/llama2.rs.git
```

2. Is to run

```sh
cargo run model.bin
```

### Command Line Interface

Options:

-   `-t <temperature>` - Set temperature (default: 1.0, 0 for greedy sampling)
-   `-p <topp>` - Set top-p sampling value (default: 0.9)
-   `-s <seed>` - Set random seed (default: timestamp)
-   `-n <steps>` - Number of tokens to generate (default: 256)
-   `-i "<prompt>"` - Provide an input prompt
-   `-z <tokenizer>` - Path to tokenizer file (default: tokenizer.bin)

Example:

```bash
cargo run --release model.bin -t 0.8 -p 0.9 -n 100 -i "Once upon a time in a magical forest,"
```

### Web Application

To build the WebAssembly version:

```bash
wasm-pack build --target web
```

Then serve the directory with your preferred web server and open `index.html`.

### Architecture

The project is structured into several key components:

-   **Transformer**: Core implementation of the Llama2 architecture
-   **Tokenizer**: BPE tokenization for text encoding/decoding
-   **Sampler**: Various sampling strategies for text generation
-   **WebAssembly binding**: Interface for browser execution
