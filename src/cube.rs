use burn::tensor::{Shape, Tensor, TensorData, backend::Backend};
use erno::{Color, Cube};

pub fn get_adjacency_matrix<B: Backend>(size: usize, device: &B::Device) -> Tensor<B, 2> {
    let num_stickers = 6 * size * size;
    let coords = generate_sticker_coordinates(size);
    let mut adjacency = Vec::with_capacity(num_stickers * num_stickers);

    let delta = 2.0 / size as f32;
    let threshold_sq = 1.5 * delta * delta;

    for i in 0..num_stickers {
        for j in 0..num_stickers {
            if i == j {
                adjacency.push(0.0);
                continue;
            }

            let p1 = coords[i];
            let p2 = coords[j];
            let dist_sq = (p1.0 - p2.0).powi(2) + (p1.1 - p2.1).powi(2) + (p1.2 - p2.2).powi(2);

            if dist_sq < threshold_sq {
                adjacency.push(1.0);
            } else {
                adjacency.push(0.0);
            }
        }
    }

    let shape = [num_stickers, num_stickers];
    let data = TensorData::new(adjacency, Shape::new(shape));
    Tensor::from_data(data, device)
}

fn generate_sticker_coordinates(size: usize) -> Vec<(f32, f32, f32)> {
    let mut coords = Vec::with_capacity(6 * size * size);
    let delta = 2.0 / size as f32;
    let start = -1.0 + delta / 2.0;

    let map = |i: usize| start + (i as f32) * delta;

    for r in 0..size {
        for c in 0..size {
            coords.push((map(c), 1.0, map(r)));
        }
    }

    for r in 0..size {
        for c in 0..size {
            coords.push((-1.0, -map(r), map(c)));
        }
    }

    for r in 0..size {
        for c in 0..size {
            coords.push((map(c), -map(r), 1.0));
        }
    }

    for r in 0..size {
        for c in 0..size {
            coords.push((1.0, -map(r), -map(c)));
        }
    }

    for r in 0..size {
        for c in 0..size {
            coords.push((-map(c), -map(r), -1.0));
        }
    }

    for r in 0..size {
        for c in 0..size {
            coords.push((map(c), -1.0, -map(r)));
        }
    }

    coords
}

pub fn cube_to_tensor<B: Backend>(cube: &Cube, device: &B::Device) -> Tensor<B, 2> {
    let mut data_flat: Vec<f32> = Vec::with_capacity(6 * cube.size * cube.size * 6);

    for face in &cube.faces {
        for sticker in &face.stickers {
            let idx = match sticker {
                Color::White => 0,
                Color::Orange => 1,
                Color::Green => 2,
                Color::Red => 3,
                Color::Blue => 4,
                Color::Yellow => 5,
            };

            for i in 0..6 {
                if i == idx {
                    data_flat.push(1.0);
                } else {
                    data_flat.push(0.0);
                }
            }
        }
    }

    let num_stickers = 6 * cube.size * cube.size;
    let shape = [num_stickers, 6];
    let data = TensorData::new(data_flat, Shape::new(shape));
    Tensor::from_data(data, device)
}
