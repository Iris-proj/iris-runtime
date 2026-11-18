use argh::FromArgs;
use inkwell::targets::{CodeModel, RelocMode};
use inkwell::OptimizationLevel;
use iris::builder::native::{NativeBuildConfig, NativeCompiler, NativeOutputFormat};
use iris::builder::paper::Paper;
use iris::data::bytecode::load_function;
use iris::vm::engine::IrisEngine;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

#[derive(FromArgs, PartialEq, Eq, Debug)]
#[argh(description = "Iris VM & Compiler CLI")]
struct IrisCli {
    #[argh(switch, short = 'v', description = "show version")]
    version: bool,

    #[argh(subcommand)]
    command: Option<Command>,
}

#[derive(FromArgs, PartialEq, Eq, Debug)]
#[argh(subcommand)]
enum Command {
    Build(BuildCmd),
    Run(RunCmd),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OptLevel {
    None,
    Less,
    Default,
    Aggressive,
}

impl FromStr for OptLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" | "none" => Ok(Self::None),
            "1" | "less" => Ok(Self::Less),
            "2" | "default" => Ok(Self::Default),
            "3" | "aggressive" => Ok(Self::Aggressive),
            _ => Err(format!("unknown optimization level '{s}', expected 0..=3")),
        }
    }
}

impl From<OptLevel> for OptimizationLevel {
    fn from(opt: OptLevel) -> Self {
        match opt {
            OptLevel::None => OptimizationLevel::None,
            OptLevel::Less => OptimizationLevel::Less,
            OptLevel::Default => OptimizationLevel::Default,
            OptLevel::Aggressive => OptimizationLevel::Aggressive,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmitFormat {
    Exe,
    Obj,
    Asm,
    Bc,
    Ll,
}

impl FromStr for EmitFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "exe" => Ok(Self::Exe),
            "obj" => Ok(Self::Obj),
            "asm" => Ok(Self::Asm),
            "bc" => Ok(Self::Bc),
            "ll" => Ok(Self::Ll),
            _ => Err(format!("unknown emit format '{s}', expected: exe, obj, asm, bc, ll")),
        }
    }
}

impl From<EmitFormat> for NativeOutputFormat {
    fn from(fmt: EmitFormat) -> Self {
        match fmt {
            EmitFormat::Exe => NativeOutputFormat::Executable,
            EmitFormat::Obj => NativeOutputFormat::Object,
            EmitFormat::Asm => NativeOutputFormat::Assembly,
            EmitFormat::Bc => NativeOutputFormat::Bitcode,
            EmitFormat::Ll => NativeOutputFormat::LlvmIr,
        }
    }
}

impl fmt::Display for EmitFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exe => write!(f, "exe"),
            Self::Obj => write!(f, "obj"),
            Self::Asm => write!(f, "asm"),
            Self::Bc => write!(f, "bc"),
            Self::Ll => write!(f, "ll"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetKind {
    Native,
    Vm,
    Cpu,
    Gpu,
}

impl FromStr for TargetKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "native" => Ok(Self::Native),
            "vm" => Ok(Self::Vm),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err(format!("unknown target '{s}', expected: native, vm, cpu, gpu")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelocKind {
    Pic,
    Static,
    DynamicNoPic,
}

impl FromStr for RelocKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pic" => Ok(Self::Pic),
            "static" => Ok(Self::Static),
            "dynamic-no-pic" => Ok(Self::DynamicNoPic),
            _ => Err(format!("unknown reloc model '{s}', expected: pic, static, dynamic-no-pic")),
        }
    }
}

impl From<RelocKind> for RelocMode {
    fn from(kind: RelocKind) -> Self {
        match kind {
            RelocKind::Pic => RelocMode::PIC,
            RelocKind::Static => RelocMode::Static,
            RelocKind::DynamicNoPic => RelocMode::DynamicNoPic,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodeModelKind {
    Default,
    Small,
    Kernel,
    Medium,
    Large,
}

impl FromStr for CodeModelKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "default" => Ok(Self::Default),
            "small" => Ok(Self::Small),
            "kernel" => Ok(Self::Kernel),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            _ => Err(format!("unknown code model '{s}', expected: default, small, kernel, medium, large")),
        }
    }
}

impl From<CodeModelKind> for CodeModel {
    fn from(kind: CodeModelKind) -> Self {
        match kind {
            CodeModelKind::Default => CodeModel::Default,
            CodeModelKind::Small => CodeModel::Small,
            CodeModelKind::Kernel => CodeModel::Kernel,
            CodeModelKind::Medium => CodeModel::Medium,
            CodeModelKind::Large => CodeModel::Large,
        }
    }
}

#[derive(FromArgs, PartialEq, Eq, Debug)]
#[argh(subcommand, name = "build", description = "compile Iris program to native targets")]
struct BuildCmd {
    #[argh(positional, description = "path to input .paper file")]
    input: PathBuf,

    #[argh(option, short = 'o', description = "output file path")]
    output: Option<PathBuf>,

    #[argh(
        option,
        default = "EmitFormat::Exe",
        description = "output format: exe, obj, asm, bc, ll (default: exe)"
    )]
    emit: EmitFormat,

    #[argh(
        option,
        short = 'O',
        default = "OptLevel::Default",
        description = "optimization level: 0, 1, 2, 3 (default: 2)"
    )]
    opt: OptLevel,

    #[argh(option, description = "target architecture triple (defaults to host)")]
    target_triple: Option<String>,

    #[argh(option, description = "target CPU architecture")]
    cpu: Option<String>,

    #[argh(option, description = "target CPU features list")]
    features: Option<String>,

    #[argh(option, description = "relocation mode: pic, static, dynamic-no-pic")]
    reloc: Option<RelocKind>,

    #[argh(option, description = "code model: default, small, kernel, medium, large")]
    code_model: Option<CodeModelKind>,

    #[argh(option, description = "custom linker executable name or path")]
    linker: Option<String>,

    #[argh(switch, short = 'q', description = "quiet mode, suppress progress logs")]
    quiet: bool,
}

#[derive(FromArgs, PartialEq, Eq, Debug)]
#[argh(subcommand, name = "run", description = "execute Iris program")]
struct RunCmd {
    #[argh(positional, description = "path to input file (.paper or .ic)")]
    input: PathBuf,

    #[argh(
        option,
        default = "TargetKind::Native",
        description = "execution target: native, vm, cpu, gpu (default: native)"
    )]
    target: TargetKind,

    #[argh(
        option,
        short = 'O',
        default = "OptLevel::Default",
        description = "optimization level for native/cpu: 0, 1, 2, 3 (default: 2)"
    )]
    opt: OptLevel,

    #[argh(switch, short = 'q', description = "quiet mode, suppress execution status logs")]
    quiet: bool,
}

fn determine_output_path(input: &Path, output: Option<PathBuf>, format: EmitFormat) -> PathBuf {
    if let Some(path) = output {
        return path;
    }

    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    match format {
        EmitFormat::Exe => PathBuf::from(stem.into_owned()),
        EmitFormat::Obj => PathBuf::from(format!("{stem}.o")),
        EmitFormat::Asm => PathBuf::from(format!("{stem}.s")),
        EmitFormat::Bc => PathBuf::from(format!("{stem}.bc")),
        EmitFormat::Ll => PathBuf::from(format!("{stem}.ll")),
    }
}

fn execute_build(cmd: BuildCmd) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = determine_output_path(&cmd.input, cmd.output, cmd.emit);
    let paper = Paper::load_binary(&cmd.input)
        .map_err(|e| format!("Failed to load Paper file from {:?}: {}", cmd.input, e))?;

    let context = inkwell::context::Context::create();
    let mut compiler = NativeCompiler::new(&context, &paper.name);

    compiler.compile_paper(&paper)?;
    compiler.emit_native_runtime()?;

    let native_format: NativeOutputFormat = cmd.emit.into();
    if native_format == NativeOutputFormat::Executable || native_format == NativeOutputFormat::Object {
        compiler.emit_main(&paper.name)?;
    }

    let mut config = NativeBuildConfig::default();
    config.opt_level = cmd.opt.into();
    config.target_triple = cmd.target_triple;
    config.cpu = cmd.cpu;
    config.features = cmd.features;
    config.linker = cmd.linker;

    if let Some(reloc) = cmd.reloc {
        config.reloc_mode = reloc.into();
    }
    if let Some(cm) = cmd.code_model {
        config.code_model = cm.into();
    }

    if !cmd.quiet {
        println!(
            "Building {:?} (format: {}, opt: {:?}) -> {:?}",
            cmd.input, cmd.emit, cmd.opt, output_path
        );
    }

    compiler.build(&output_path, native_format, &config)?;

    if !cmd.quiet {
        println!("Successfully built: {:?}", output_path);
    }

    Ok(())
}

fn execute_run(cmd: RunCmd) -> Result<(), Box<dyn std::error::Error>> {
    if cmd.input.extension().and_then(|s| s.to_str()) == Some("ic") {
        let func = load_function(
            cmd.input
                .to_str()
                .ok_or("Invalid non-UTF8 input path string")?,
        )?;
        let mut engine = IrisEngine::new();
        engine.push_frame(std::rc::Rc::new(func), 0)?;
        engine.run()?;
        if let Some(val) = engine.stack.pop() {
            println!("Result: {val:?}");
        }
        return Ok(());
    }

    let paper = Paper::load_binary(&cmd.input)
        .map_err(|e| format!("Failed to load Paper file from {:?}: {}", cmd.input, e))?;

    let opt_level = cmd.opt.into();

    match cmd.target {
        TargetKind::Native => {
            let (code, stdout) = paper.run_native(opt_level)?;
            if !stdout.is_empty() {
                print!("{stdout}");
            }
            if !cmd.quiet || code != 0 {
                println!("Process exited with code: {code}");
            }
        }
        TargetKind::Vm => {
            let mut engine = IrisEngine::new();
            let res = paper.run_vm(&mut engine)?;
            println!("VM Result: {res:?}");
        }
        TargetKind::Cpu => {
            let context = inkwell::context::Context::create();
            let module = context.create_module(&paper.name);
            let rfc = iris::builder::compiler::RuntimeFunctionCompiler::new(&context, module);
            let ee = rfc
                .module
                .create_jit_execution_engine(opt_level)
                .map_err(|e| format!("JIT Engine error: {e:?}"))?;
            rfc.setup_ee(&ee);
            let res = paper.run_cpu(&rfc, &ee)?;
            println!("CPU JIT Result: {res:?}");
        }
        TargetKind::Gpu => {
            let mut gpu = iris::vm::gpu::GpuDevice::new();
            let res = paper.run_gpu(&mut gpu)?;
            println!("GPU Result: {:?}", res.return_value);
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && (args[1] == "--version" || args[1] == "-V") {
        println!("iris {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let cli: IrisCli = argh::from_env();
    if cli.version {
        println!("iris {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    match cli.command {
        Some(Command::Build(build)) => execute_build(build),
        Some(Command::Run(run)) => execute_run(run),
        None => {
            println!("Iris VM & Compiler v{}", env!("CARGO_PKG_VERSION"));
            println!("Usage: iris <command> [<args>]\n");
            println!("Commands:\n  build    compile Iris program to native targets\n  run      execute Iris program\n");
            println!("Run 'iris --help' for full details.");
            Ok(())
        }
    }
}
