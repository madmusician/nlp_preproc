use std;
use std::sync::atomic::{AtomicUsize, Ordering};

// Ad-hoc add increment to AtomicUsize.
pub trait AtomicInc {
    fn inc(&self);
}

impl AtomicInc for AtomicUsize {
    fn inc(&self) {
        self.fetch_add(1, Ordering::Relaxed);
    }
}

// Ad-hoc add "store the max value from given value and current value" to AtomicUsize
pub trait AtomicStoreMax {
    type Inner;
    fn store_max(&self, x: Self::Inner);
}

impl AtomicStoreMax for AtomicUsize {
    type Inner = usize;
    fn store_max(&self, x: usize) {
        loop {
            let cur = self.load(Ordering::Relaxed);
            if x <= cur {
                break;
            }
            if self.compare_exchange(cur, x, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

/// Saving an nested Vec into an npy file.
/// Very limited, currently only work for array of `u32` and `f32`,
/// and atmost 3-D array is permitted.
///
/// # Examples
/// ```
/// let array = vec![vec![1u32, 2u32], vec![3u32]];
/// save_npy_with_iter("a.npy", &array
/// ```
pub fn save_npy_with_iter<P: AsRef<std::path::Path>, I: WriteNdVec, T: IntoIterator<Item = I>>(
    path: P,
    data: T,
    target_shape: <I::Shape as NdShape>::OuterShape,
    allow_padding: bool,
) -> Result<(), std::io::Error> {
    use byteorder::{WriteBytesExt, LE};
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let mut fp = BufWriter::new(File::create(path)?);

    fp.write(&[0x93])?;
    fp.write("NUMPY\x01\x00".as_bytes())?;

    // `{:?}` for target shape is enough
    let mut header_str = format!(
        r##"{{"descr": "{}","fortran_order": False,"shape": {:?} }}"##,
        I::DType::NP_TYPE_STRING,
        target_shape
    );
    let num_bytes_to_fill = (header_str.len() + 10 + 15) / 16 * 16 - (header_str.len() + 10);
    header_str.extend(std::iter::repeat(' ').take(num_bytes_to_fill - 1));
    header_str.push('\n');

    fp.write_u16::<LE>(header_str.len() as u16)?;
    fp.write(header_str.as_bytes())?;

    let mut num_rows_written = 0;
    for row in data.into_iter() {
        assert!(num_rows_written < target_shape.first_dim());
        row.write_with_shape(&mut fp, target_shape.tail_dims(), allow_padding)?;
        num_rows_written += 1;
    }
    let mut zero_row = None;
    for _ in num_rows_written..target_shape.first_dim() {
        let zero_row = zero_row
            .get_or_insert_with(|| vec![I::DType::default(); target_shape.tail_dims().product()]);
        (*zero_row).write_with_shape(
            &mut fp,
            (target_shape.tail_dims().product(),),
            allow_padding,
        )?; // write flattened array
    }
    Ok(())
}

pub trait WriteNdVec: Clone {
    type Shape: NdShape;
    // Enough for tuples
    type DType: NumpyDtypeStr + Default + Clone + WriteNdVec<Shape = ()>;
    fn write_with_shape<W: std::io::Write>(
        &self,
        writer: &mut W,
        shape: Self::Shape,
        allow_padding: bool,
    ) -> std::io::Result<()>;
}

impl<'a, T: WriteNdVec> WriteNdVec for &'a T {
    type Shape = T::Shape;
    type DType = T::DType;
    fn write_with_shape<W: std::io::Write>(
        &self,
        writer: &mut W,
        shape: Self::Shape,
        allow_padding: bool,
    ) -> std::io::Result<()> {
        (*self).write_with_shape(writer, shape, allow_padding)
    }
}

impl WriteNdVec for u32 {
    type Shape = ();
    type DType = u32;
    fn write_with_shape<W: std::io::Write>(
        &self,
        writer: &mut W,
        _shape: Self::Shape,
        _allow_padding: bool,
    ) -> std::io::Result<()> {
        writer.write(slice_to_bytes(&[*self]))?;
        Ok(())
    }
}

impl WriteNdVec for f32 {
    type Shape = ();
    type DType = f32;
    fn write_with_shape<W: std::io::Write>(
        &self,
        writer: &mut W,
        _shape: Self::Shape,
        _allow_padding: bool,
    ) -> std::io::Result<()> {
        writer.write(slice_to_bytes(&[*self]))?;
        Ok(())
    }
}

impl<T: WriteNdVec + Clone> WriteNdVec for Vec<T> {
    type Shape = <<T as WriteNdVec>::Shape as NdShape>::OuterShape;
    type DType = <T as WriteNdVec>::DType;
    fn write_with_shape<W: std::io::Write>(
        &self,
        writer: &mut W,
        shape: Self::Shape,
        allow_padding: bool,
    ) -> std::io::Result<()> {
        assert!(
            allow_padding && self.len() == shape.first_dim() || self.len() <= shape.first_dim()
        );
        for row in self.iter() {
            row.write_with_shape(writer, shape.tail_dims(), allow_padding)?;
        }
        let mut zero_row = None;
        for _ in self.len()..shape.first_dim() {
            let zero_row = zero_row
                .get_or_insert_with(|| vec![Self::DType::default(); shape.tail_dims().product()]);
            (*zero_row).write_with_shape(writer, (shape.tail_dims().product(),), allow_padding)?; // write flattened array
        }
        Ok(())
    }
}

fn slice_to_bytes<'a, T: Sized>(s: &'a [T]) -> &'a [u8] {
    unsafe {
        std::slice::from_raw_parts(
            s.as_ptr() as *const u8,
            s.len() * std::mem::size_of::<T>() / std::mem::size_of::<u8>(),
        )
    }
}

pub trait NumpyDtypeStr {
    const NP_TYPE_STRING: &'static str;
}

impl NumpyDtypeStr for u32 {
    const NP_TYPE_STRING: &'static str = "<u4";
}

impl NumpyDtypeStr for f32 {
    const NP_TYPE_STRING: &'static str = "<f4";
}

pub trait NdShape: Copy + std::fmt::Debug {
    // Debug is enough for directly convert to python-like string
    type InnerShape: NdShape;
    /// If this shape is a slice of a larger tensor's shape, then what is the type of that shape?
    type OuterShape: NdShape<InnerShape = Self>;
    fn first_dim(self) -> usize;
    fn tail_dims(self) -> Self::InnerShape;
    fn product(self) -> usize;
}

impl NdShape for () {
    type InnerShape = NdShapeHighDimTermination;
    // In fact unused, for forming loop
    type OuterShape = (usize,);
    fn first_dim(self) -> usize {
        unreachable!()
    }
    fn tail_dims(self) -> Self::InnerShape {
        unreachable!()
    }
    fn product(self) -> usize {
        1 // for usize, f32, ...
    }
}

impl NdShape for (usize,) {
    type InnerShape = ();
    type OuterShape = (usize, usize);
    fn first_dim(self) -> usize {
        self.0
    }
    fn tail_dims(self) {}
    fn product(self) -> usize {
        self.0
    }
}

impl NdShape for (usize, usize) {
    type InnerShape = (usize,);
    type OuterShape = (usize, usize, usize);
    fn first_dim(self) -> usize {
        self.0
    }
    fn tail_dims(self) -> (usize,) {
        (self.1,)
    }
    fn product(self) -> usize {
        self.0 * self.1
    }
}

impl NdShape for (usize, usize, usize) {
    type InnerShape = (usize, usize);
    type OuterShape = NdShapeHighDimTermination;
    fn first_dim(self) -> usize {
        self.0
    }
    fn tail_dims(self) -> Self::InnerShape {
        (self.1, self.2)
    }
    fn product(self) -> usize {
        self.0 * self.1 * self.2
    }
}

// for termination of NdShape impl, DON'T USE!
#[derive(Debug, Copy, Clone)]
#[doc(hidden)]
pub struct NdShapeHighDimTermination;

impl NdShape for NdShapeHighDimTermination {
    type InnerShape = (usize, usize, usize);
    type OuterShape = ();
    fn first_dim(self) -> usize {
        unreachable!()
    }
    fn tail_dims(self) -> Self::InnerShape {
        unreachable!()
    }
    fn product(self) -> usize {
        unreachable!()
    }
}

#[test]
fn test_save_npy() {
    let array = vec![vec![1, 2, 3], vec![4, 5, 6]];
    use mktemp::Temp;
    let temp = Temp::new_file().unwrap();
    let escaped = temp.as_ref().to_string_lossy().replace("\\", "\\\\");
    save_npy_with_iter(&temp, &array, (2, 3), false).unwrap();
    let status = std::process::Command::new("python")
        .arg("-c")
        .arg(format!(
            r##"import numpy;a=numpy.load("{}");print a"##,
            escaped
        ))
        .status()
        .unwrap();
    assert!(status.success());
}
