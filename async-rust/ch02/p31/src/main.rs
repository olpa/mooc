use std::ptr;

struct SelfReferential {
    data: String,
    self_pointer: *const String,
}

impl SelfReferential {
    fn new(data: String) -> SelfReferential {
        let mut sr = SelfReferential {
            data,
            self_pointer: ptr::null(),
        };
        sr.self_pointer = &sr.data as *const String;
        sr.print();
        eprintln!("It was a constructor, {:p}", sr.self_pointer);
        sr
    }
    fn print(&self) {
        unsafe {
            eprintln!("As str itself: {}", self.data);
            eprintln!("As current ptr: {:p}", &self.data);
            eprintln!("As ptr: {}", *self.self_pointer);
        }
    }
}

fn main() {
    let s = "first".to_string();
    let f = SelfReferential::new(s);
    eprintln!("Now in main, {:p}", f.self_pointer);
    f.print();
    //let mf = f;
    //mf.print();
}

