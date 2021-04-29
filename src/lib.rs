use ndarray::prelude::*;
pub use num_complex::Complex;
use num_traits::{Num, Zero};
use rayon::prelude::*;
use rustfft::FftNum;
pub use rustfft::FftPlannerAvx;
use std::cmp::max;
use transpose::transpose_inplace;

/// Getting the norm of each elements
pub fn abs(m: &Vec<Vec<Complex<f64>>>) -> Vec<Vec<f64>> {
  m.iter()
    .map(|row| row.iter().map(|el| el.norm()).collect())
    .collect()
}

/// Square each elements
pub fn square<T: Num + Copy>(m: &Vec<Vec<T>>) -> Vec<Vec<T>> {
  m.iter()
    .map(|row| row.iter().map(|el| el.mul(*el)).collect())
    .collect()
}

/// Fast fourier transform 2d
pub fn fft2<T: FftNum>(input_matrix: &Vec<Vec<Complex<T>>>) -> Vec<Vec<Complex<T>>> {
  let mut planer = FftPlannerAvx::new().unwrap();
  let mut input_tp = tp(&input_matrix);
  let fft = planer.plan_fft_forward(input_tp[0].len());
  input_tp.iter_mut().for_each(|mut row| {
    fft.process(&mut row);
  });
  let mut input_tp_tp = tp(&input_tp);
  planer.plan_fft_forward(input_tp_tp[0].len());
  input_tp_tp.iter_mut().for_each(|mut row| {
    fft.process(&mut row);
  });
  input_tp_tp
}

/// Fast inverse fourier transform 2d
pub fn ifft2<T: FftNum>(input_matrix: &Vec<Vec<Complex<T>>>) -> Vec<Vec<Complex<T>>> {
  let mut input_tp = tp(&input_matrix);
  let mut planer = FftPlannerAvx::new().unwrap();
  let fft = planer.plan_fft_inverse(input_tp[0].len());
  input_tp.par_iter_mut().for_each(|mut row| {
    fft.process(&mut row);
    let row_length = row.len();
    row.iter_mut().for_each(|el| {
      el.re = el.re / T::from_usize(row_length).expect("Can't convert usize to this type");
      el.im = el.im / T::from_usize(row_length).expect("Can't convert usize to this type");
    });
  });
  let mut input_tp_tp = tp(&input_tp);
  planer.plan_fft_inverse(input_tp_tp[0].len());
  input_tp_tp.par_iter_mut().for_each(|mut row| {
    fft.process(&mut row);
    let row_length = row.len();
    row.iter_mut().for_each(|el| {
      el.re = el.re / T::from_usize(row_length).expect("Can't convert usize to this type");
      el.im = el.im / T::from_usize(row_length).expect("Can't convert usize to this type");
    });
  });
  input_tp_tp
}

/// Get conj of each elements
pub fn conj(i: Vec<Vec<Complex<f64>>>) -> Vec<Vec<Complex<f64>>> {
  i.iter()
    .map(|row| row.iter().map(|el| el.conj()).collect())
    .collect()
}

/// Rotate the matrix counter clockwise 180 degrees
pub fn rot180<T: Clone>(i: &Vec<Vec<T>>) -> Vec<Vec<T>> {
  let mut new_matrix = i.clone();
  new_matrix.reverse();
  new_matrix.iter_mut().for_each(|row| row.reverse());
  new_matrix
}

/// Rotate the matrix counter clockwise 180 degrees inplace
pub fn rot180_inplace<T>(i: &mut Vec<Vec<T>>) {
  i.reverse();
  i.iter_mut().for_each(|row| row.reverse());
}

/// Elementwise multiply then sum
pub fn conv<T: Num + Copy>(a: &Vec<Vec<Complex<T>>>, b: &Vec<Vec<Complex<T>>>) -> Complex<T> {
  a.iter()
    .zip(b.iter())
    .map(|(row_a, row_b)| {
      row_a
        .iter()
        .zip(row_b.iter())
        .map(|(a_el, b_el)| a_el * b_el)
        .sum::<Complex<T>>()
    })
    .sum()
}

/// 2d full convolution with fft2 and ifft2 boosted
/// Refer to @Royi's answer in https://stackoverflow.com/questions/50614085/applying-low-pass-and-laplace-of-gaussian-filter-in-frequency-domain
/// and @ZR Han's point out on https://stackoverflow.com/questions/12253984/linear-convolution-of-two-images-in-matlab-using-fft2
pub fn conv2_fft<T: FftNum>(
  picture: &Vec<Vec<Complex<T>>>,
  kernel: &Vec<Vec<Complex<T>>>,
) -> Vec<Vec<Complex<T>>> {
  let picture_width = picture[0].len();
  let picture_height = picture.len();
  let kernel_width = kernel[0].len();
  let kernel_height = kernel.len();

  ifft2(
    &fft2(&zero_pad(&picture, kernel_width - 1, kernel_height - 1))
      .par_iter()
      .zip(fft2(&zero_pad(&kernel, picture_width - 1, picture_height - 1)).par_iter())
      .map(|(picture_row, kernel_row)| {
        picture_row
          .iter()
          .zip(kernel_row.iter())
          .map(|(i_el, k_el)| i_el * k_el)
          .collect::<Vec<Complex<T>>>()
      })
      .collect::<Vec<Vec<Complex<T>>>>(),
  )
}

/// 2d full convolution with just hard matrix multiply
pub fn conv2<T: Num + Zero + Copy>(picture: &Vec<Vec<T>>, kernel: &Vec<Vec<T>>) -> Vec<Vec<T>> {
  let picture_width = picture[0].len();
  let picture_height = picture.len();
  let kernel_width = kernel[0].len();
  let kernel_height = kernel.len();

  let conv_workspace_width = picture_width + kernel_width * 2 - 2;
  let conv_workspace_height = picture_height + kernel_height * 2 - 2;

  // make a zero padded workspace matrix
  let mut conv_workspace_arr = Array2::<T>::zeros((conv_workspace_height, conv_workspace_width));

  // copy the image to the center of the workspace
  for (idx, row) in picture.iter().enumerate() {
    for (col_idx, el) in row.iter().enumerate() {
      conv_workspace_arr[[idx + kernel_height - 1, col_idx + kernel_width - 1]] = el.clone();
    }
  }

  let mut kernel_arr = Array2::<T>::zeros((kernel_height, kernel_width));
  kernel.iter().enumerate().for_each(|(row_idx, row)| {
    row.iter().enumerate().for_each(|(col_idx, el)| {
      kernel_arr[[row_idx, col_idx]] = el.clone();
    });
  });

  let result_flat = conv_workspace_arr
    .windows((kernel_height, kernel_width))
    .into_iter()
    .map(|window| (&window * &kernel_arr).sum())
    .collect::<Vec<T>>();
  result_flat
    .chunks(picture_width + kernel_width - 1)
    .map(|chunk| chunk.to_vec())
    .collect()
}

/// pretty print matrix
pub fn pretty_print<T: std::fmt::Display + std::fmt::LowerExp>(m: &Vec<Vec<T>>) {
  m.iter().for_each(|row| {
    print!("[");
    row.iter().for_each(|el| print!("{:10.3e}, ", el));
    print!("]\n");
  });
  let width = {
    if m.is_empty() {
      0
    } else {
      m[0].len()
    }
  };
  println!("{}x{}", m.len(), width);
}

/// Transpose with clone
pub fn tp<T: Zero + Copy>(m: &Vec<Vec<T>>) -> Vec<Vec<T>> {
  let input_matrix_width = m[0].len();
  let input_matrix_height = m.len();
  let mut m_flat: Vec<T> = m.iter().flatten().cloned().collect();
  let mut scratch = vec![T::zero(); max(input_matrix_width, input_matrix_height)];
  transpose_inplace(
    &mut m_flat,
    &mut scratch,
    input_matrix_width,
    input_matrix_height,
  );
  m_flat
    .chunks(input_matrix_height)
    .map(|chunk| chunk.to_vec())
    .collect::<Vec<Vec<T>>>()
}

/// Pad matrix with zeros to the bottom and right
pub fn zero_pad<T: Zero + Clone>(
  origin: &Vec<Vec<T>>,
  pad_width: usize,
  pad_height: usize,
) -> Vec<Vec<T>> {
  let origin_height = origin.len();
  let origin_width = origin[0].len();
  let mut padded = vec![vec![T::zero(); origin_width + pad_width]; origin_height + pad_height];
  // copy the origin matrix to the topleft
  for (row_idx, row) in origin.iter().enumerate() {
    for (col_idx, el) in row.iter().enumerate() {
      padded[row_idx][col_idx] = el.clone();
    }
  }
  padded
}

mod test {
  use super::*;
  #[test]
  fn test_conv2() {
    let x = vec![
      vec![
        Complex::new(8.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(6.0, 0.0),
      ],
      vec![
        Complex::new(3.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(7.0, 0.0),
      ],
      vec![
        Complex::new(4.0, 0.0),
        Complex::new(9.0, 0.0),
        Complex::new(2.0, 0.0),
      ],
    ];
    let y = vec![
      vec![Complex::new(1.0, 0.0), Complex::new(3.0, 0.0)],
      vec![Complex::new(4.0, 0.0), Complex::new(2.0, 0.0)],
    ];
    pretty_print(&conv2(&x, &y));
    pretty_print(&conv2_fft(&x, &rot180(&y)));
  }
  #[test]
  fn test_fft_ifft() {
    let signal = vec![
      vec![
        Complex::new(8.0, 0.0),
        Complex::new(1.0, 0.0),
        Complex::new(6.0, 0.0),
      ],
      vec![
        Complex::new(3.0, 0.0),
        Complex::new(5.0, 0.0),
        Complex::new(7.0, 0.0),
      ],
      vec![
        Complex::new(4.0, 0.0),
        Complex::new(9.0, 0.0),
        Complex::new(2.0, 0.0),
      ],
    ];
    let signal2 = vec![vec![
      Complex::new(1.0, 0.0),
      Complex::new(2.0, 0.0),
      Complex::new(3.0, 0.0),
      Complex::new(4.0, 0.0),
      Complex::new(5.0, 0.0),
    ]];
    pretty_print(&signal);
    pretty_print(&fft2(&signal));
    pretty_print(&ifft2(&fft2(&signal)));
    pretty_print(&signal2);
    pretty_print(&fft2(&signal2));
    pretty_print(&ifft2(&fft2(&signal2)));
  }
}
