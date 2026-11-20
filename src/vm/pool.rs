use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex, OnceLock, RwLock};
use std::thread;
use crate::builder::paper::Paper;

type Job = Box<dyn FnOnce() + Send + 'static>;


pub struct IrisThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Job>>,
}

struct Worker {
    #[allow(dead_code)]
    pub id: usize,
    pub thread: Option<thread::JoinHandle<()>>,
}

static GLOBAL_POOL: OnceLock<IrisThreadPool> = OnceLock::new();
static GLOBAL_PAPERS: OnceLock<RwLock<HashMap<String, Paper>>> = OnceLock::new();


pub fn register_global_paper(paper: Paper) {
    let registry = GLOBAL_PAPERS.get_or_init(|| RwLock::new(HashMap::new()));
    registry.write().unwrap().insert(paper.name.clone(), paper);
}


pub fn get_global_paper(name: &str) -> Option<Paper> {
    let registry = GLOBAL_PAPERS.get_or_init(|| RwLock::new(HashMap::new()));
    registry.read().unwrap().get(name).cloned()
}

impl IrisThreadPool {

    pub fn global() -> &'static IrisThreadPool {
        GLOBAL_POOL.get_or_init(|| {
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            IrisThreadPool::new(cores)
        })
    }


    pub fn new(size: usize) -> Self {
        assert!(size > 0);
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));

        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            let rx = Arc::clone(&receiver);
            let thread = thread::Builder::new()
                .name(format!("iris-worker-{}", id))
                .spawn(move || {
                    while let Ok(job) = rx.lock().unwrap().recv() {
                        job();
                    }
                })
                .expect("Failed to spawn Iris thread pool worker");
            workers.push(Worker {
                id,
                thread: Some(thread),
            });
        }

        IrisThreadPool {
            workers,
            sender: Some(sender),
        }
    }


    #[inline]
    pub fn spawn<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        if let Some(ref sender) = self.sender {
            let _ = sender.send(Box::new(f));
        }
    }
}

impl Drop for IrisThreadPool {
    fn drop(&mut self) {
        drop(self.sender.take());
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}
