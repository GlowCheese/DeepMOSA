# DeepMOSA: Empowering Test Generation through LLM-Augmented Search


DeepMOSA is an automated unit testing method that leverages Large Language Model (LLMs) as a search assistant to improve test generation. Built upon **CodaMOSA** and **DynaMOSA**, our approach introduces several key enhancements to address current limitations and further boost branch coverage:

- **Asynchronous Search**: Allows the evolutionary algorithm and LLM querying to operate in parallel, reducing bottlenecks and improving search efficiency.  
- **Context-Aware and Branch-Targeted Prompts**: Guides the LLM to generate more relevant test cases by leveraging execution context and explicit branch objectives.
- **Improved Test Case Deserialization**: Improves robustness and accuracy of reconstructed test inputs from LLM outputs.

By combining evolutionary search with LLM-augmented guidance, DeepMOSA provides a more adaptive and effective method for automated unit test generation.


## Installation

### Prerequisites
- [uv package manager](https://docs.astral.sh/uv)  

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/GlowCheese/deepmosa
   cd deepmosa
    ```

2. Create a `.env` file in the project root to configure model access:
   ```bash
   # Acknowledge potential risks of automatically generated tests
   PYNGUIN_DANGER_AWARE=true

   # Example: Using DeepSeek model
   DEFAULT_MODEL=DEEPSEEK
   DEEPSEEK_API_KEY=your_deepseek_api_key_here

   # Example: Using ChatGPT model
   DEFAULT_MODEL=CHATGPT
   CHATGPT_API_KEY=your_chatgpt_api_key_here
   ```

3. Install the CLI using `uv`:
   ```bash
   uv tool install -e . --python 3.10
   ```



## Usage

Once installed, DeepMOSA can be invoked through the Pynguin CLI.  
Here is a basic example:

```bash
pynguin \
  --project-path ~/project/example \
  --module-name example.module \
  --maximum-search-time 60 \
  --algorithm DEEPMOSA \
  --allow-expandable-cluster \
  --async-enabled
```

**Explanation:**

* `--project-path`: Path to the target project directory.
* `--module-name`: Python module under test.
* `--maximum-search-time`: Maximum duration of the search process (in seconds).
* `--algorithm`: The test generation algorithm. Use `DEEPMOSA` to enable our method.
* `--allow-expandable-cluster` *(introduced in CodaMOSA)*:
  Allows callable elements seen in LLM responses to be added to the test cluster for further exploration.
* `--async-enabled` *(introduced in DeepMOSA)*:
  Enables asynchronous execution, allowing the evolutionary process and LLM querying to run in parallel.

**Results:**

_coming soon_


## License

DeepMOSA is licensed under the [MIT License](LICENSE).  
It builds on Pynguin v0.40.0, which has used MIT since v0.30.0 (previously LGPL).
