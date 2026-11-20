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

#[allow(dead_code)]
mod style {
    use std::io::IsTerminal;

    fn is_color_enabled() -> bool {
        if std::env::var_os("NO_COLOR").is_some() {
            return false;
        }
        if std::env::var("TERM").map(|t| t == "dumb").unwrap_or(false) {
            return false;
        }
        std::io::stdout().is_terminal()
    }

    pub fn bold(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[1m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn dim(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[2m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn cyan(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn bold_cyan(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[1;36m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn green(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn bold_green(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[1;32m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn yellow(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[33m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn bold_yellow(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[1;33m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn red(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[31m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn bold_red(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[1;31m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn magenta(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[35m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn bold_magenta(s: impl std::fmt::Display) -> String {
        if is_color_enabled() {
            format!("\x1b[1;35m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn colorize_help(text: &str) -> String {

        let text = text.replace("--help, help", "-h, --help, help");
        if !is_color_enabled() {
            return text;
        }

        let mut out = String::with_capacity(text.len() + 512);
        for line in text.lines() {
            if line.starts_with("Usage: ") {
                let rest = &line[7..];
                out.push_str(&bold("Usage: "));
                let mut first = true;
                for part in rest.split_whitespace() {
                    if !first {
                        out.push(' ');
                    }
                    if first {
                        out.push_str(&bold_cyan(part));
                        first = false;
                    } else if part.starts_with('-') {
                        out.push_str(&yellow(part));
                    } else if (part.starts_with('<') && part.ends_with('>'))
                        || (part.contains('<') && part.contains('>'))
                    {
                        out.push_str(&cyan(part));
                    } else if part.starts_with('[') || part.starts_with(']') {
                        out.push_str(&dim(part));
                    } else {
                        out.push_str(&green(part));
                    }
                }
                out.push('\n');
            } else if line.starts_with("Required positional") {
                out.push_str(&bold_red(line));
                out.push('\n');
            } else if line.ends_with(':') && !line.starts_with(' ') {
                out.push_str(&bold_yellow(line));
                out.push('\n');
            } else if line.starts_with("  -") {
                let trimmed = line.trim_start();
                let indent = &line[..line.len() - trimmed.len()];
                if let Some(idx) = trimmed.find("  ") {
                    let flag_part = &trimmed[..idx];
                    let desc = &trimmed[idx..];
                    out.push_str(indent);
                    out.push_str(&yellow(flag_part));
                    out.push_str(&dim(desc));
                } else {
                    out.push_str(indent);
                    out.push_str(&yellow(trimmed));
                }
                out.push('\n');
            } else if line.starts_with("  ") && !line.starts_with("    ") {
                let trimmed = line.trim_start();
                let indent = &line[..line.len() - trimmed.len()];
                if let Some(idx) = trimmed.find("  ") {
                    let cmd_part = &trimmed[..idx];
                    let desc = &trimmed[idx..];
                    out.push_str(indent);
                    out.push_str(&green(cmd_part));
                    out.push_str(&dim(desc));
                } else {
                    out.push_str(line);
                }
                out.push('\n');
            } else if line.starts_with("    ") && !line.trim().is_empty() {
                out.push_str("    ");
                out.push_str(&yellow(line.trim()));
                out.push('\n');
            } else if line.starts_with("Unrecognized") || line.starts_with("Error") {
                out.push_str(&bold_red(line));
                out.push('\n');
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }
}

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

    #[argh(switch, description = "pure AOT compilation without VM engine structs, VM calls, or global mappings")]
    pure: bool,

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

    let native_format: NativeOutputFormat = cmd.emit.into();
    if cmd.pure {
        compiler.compile_paper_pure(&paper)?;
        if native_format == NativeOutputFormat::Executable || native_format == NativeOutputFormat::Object {
            compiler.emit_main_pure(&paper.name)?;
        }
    } else {
        compiler.compile_paper(&paper)?;
        compiler.emit_native_runtime()?;
        if native_format == NativeOutputFormat::Executable || native_format == NativeOutputFormat::Object {
            compiler.emit_main(&paper.name)?;
        }
    }

    let mut config = NativeBuildConfig::default();
    config.opt_level = cmd.opt.into();
    config.target_triple = cmd.target_triple;
    config.cpu = cmd.cpu;
    config.features = cmd.features;
    config.linker = cmd.linker;
    config.pure_aot = cmd.pure;

    if let Some(reloc) = cmd.reloc {
        config.reloc_mode = reloc.into();
    }
    if let Some(cm) = cmd.code_model {
        config.code_model = cm.into();
    }

    if !cmd.quiet {
        println!(
            "{} {:?} ({} {}, {} {:?}) -> {}",
            style::bold_cyan("Building"),
            cmd.input,
            style::dim("format:"),
            style::yellow(cmd.emit),
            style::dim("opt:"),
            style::yellow(format!("{:?}", cmd.opt)),
            style::bold_green(output_path.display())
        );
    }

    compiler.build(&output_path, native_format, &config)?;

    if !cmd.quiet {
        println!(
            "{} {}",
            style::bold_green("Successfully built:"),
            style::bold(output_path.display())
        );
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
            println!("{} {val:?}", style::bold_green("Result:"));
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
                if code == 0 {
                    println!(
                        "{} {}",
                        style::dim("Process exited with code:"),
                        style::green(code)
                    );
                } else {
                    eprintln!(
                        "{} {}",
                        style::bold_red("Process exited with code:"),
                        style::bold_red(code)
                    );
                }
            }
        }
        TargetKind::Vm => {
            let mut engine = IrisEngine::new();
            let res = paper.run_vm(&mut engine)?;
            println!("{} {res:?}", style::bold_cyan("VM Result:"));
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
            println!("{} {res:?}", style::bold_magenta("CPU JIT Result:"));
        }
        TargetKind::Gpu => {
            let mut gpu = iris::vm::gpu::GpuDevice::new();
            let res = paper.run_gpu(&mut gpu)?;
            println!("{} {:?}", style::bold_yellow("GPU Result:"), res.return_value);
        }
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && (args[1] == "--version" || args[1] == "-V") {
        println!("{} {}", style::bold_cyan("iris"), env!("CARGO_PKG_VERSION"));
        return;
    }

    let cmd_name = args.first().map(|s| s.as_str()).unwrap_or("iris");

    let normalized_args: Vec<String> = args
        .iter()
        .skip(1)
        .map(|arg| {
            if arg == "-h" {
                "--help".to_string()
            } else {
                arg.clone()
            }
        })
        .collect();
    let cmd_slices: Vec<&str> = normalized_args.iter().map(|s| s.as_str()).collect();

    let cli = match IrisCli::from_args(&[cmd_name], &cmd_slices) {
        Ok(cli) => cli,
        Err(early_exit) => {
            let colored = style::colorize_help(&early_exit.output);
            if early_exit.status.is_ok() {
                print!("{colored}");
                std::process::exit(0);
            } else {
                eprint!("{colored}");
                std::process::exit(1);
            }
        }
    };

    if cli.version {
        println!("{} {}", style::bold_cyan("iris"), env!("CARGO_PKG_VERSION"));
        return;
    }

    let res = match cli.command {
        Some(Command::Build(build)) => execute_build(build),
        Some(Command::Run(run)) => execute_run(run),
        None => {
            println!(
                "{} v{}",
                style::bold_cyan("Iris VM & Compiler"),
                env!("CARGO_PKG_VERSION")
            );
            println!(
                "{}: {} <command> [<args>]\n",
                style::bold("Usage"),
                style::cyan("iris")
            );
            println!(
                "{}:\n  {}    {}\n  {}      {}\n",
                style::bold("Commands"),
                style::green("build"),
                style::dim("compile Iris program to native targets"),
                style::green("run"),
                style::dim("execute Iris program")
            );
            println!(
                "Run '{}' for full details.",
                style::yellow("iris --help")
            );
            Ok(())
        }
    };

    if let Err(err) = res {
        eprintln!("{} {err}", style::bold_red("error:"));
        std::process::exit(1);
    }
}
