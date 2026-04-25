<img width="1500" height="500" alt="IrisRuntimeB" src="https://github.com/user-attachments/assets/a8dba5ea-0c17-4703-bebe-3113191aafe2" />

# Iris Runtime: A High-Performance Execution Environment

Iris is a lightweight, fast, and embeddable runtime environment. What began as a standalone virtual machine has evolved into a comprehensive runtime for the Iris ecosystem, providing tools for building, managing, and executing Iris bytecode.

## About the Iris Project

The Iris project provides a modular framework for language development. By offering a unified **Runtime**, **VM**, and **Toolchain**, it enables an ecosystem similar to .NET or the JVM, where multiple languages can interoperate seamlessly on a shared infrastructure.

## Key Components

* **Iris VM:** The core execution engine optimized for high-performance bytecode interpretation.
* **Iris Builder:** A dedicated toolchain for compiling and packaging Iris applications.
* **Standard Library:** Built-in support for object-oriented primitives, including classes, instances, and methods.

## Features

* **Integrated Runtime:** A complete environment that handles everything from bytecode loading to memory management.
* **Rust-Powered Performance:** Built with Rust for memory safety and zero-cost abstractions, leveraging LLVM for efficient IR generation.
* **Embeddable Design:** Designed to be integrated into larger applications as a scripting or execution engine.
* **Cross-Platform:** Native support for various architectures and operating systems.

## Getting Started

To build the Iris Runtime and its associated tools, ensure you have the Rust toolchain installed:

```bash
git clone [https://github.com/Iris-proj/iris-vm.git](https://github.com/Iris-proj/iris-runtime.git)
cargo build
```

## Contributing

Contributions are welcome! If you'd like to contribute to the project, please fork the repository and submit a pull request.

## License

Iris VM is licensed under the GPL License. See the [LICENSE](LICENSE) file for more information.
