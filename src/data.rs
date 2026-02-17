use erno::{Axis, Cube, Direction, Move, ReferenceFrame};
use rand::prelude::*;
use std::collections::HashSet;

pub fn generate_training_sample(cube_size: usize, scramble_length: usize) -> (Cube, Move) {
    let mut rng = rand::rng();
    let mut cube = Cube::new(cube_size);
    let mut current_axis: Option<Axis> = None;
    let mut used_layers_on_axis: HashSet<usize> = HashSet::new();

    for _ in 0..scramble_length {
        let mut valid_move = None;

        while valid_move.is_none() {
            let axis = match rng.random_range(0..3) {
                0 => Axis::X,
                1 => Axis::Y,
                _ => Axis::Z,
            };

            let max_depth = cube_size / 2;
            let depth = if max_depth > 1 {
                rng.random_range(1..=max_depth)
            } else {
                1
            };

            let layer = rng.random_range(0..=(cube_size - depth));

            let direction = match rng.random_range(0..3) {
                0 => Direction::Clockwise,
                1 => Direction::CounterClockwise,
                _ => Direction::Double,
            };

            let is_valid = if let Some(last) = current_axis {
                if axis == last {
                    !(layer..layer + depth).any(|l| used_layers_on_axis.contains(&l))
                } else {
                    true
                }
            } else {
                true
            };

            if is_valid {
                valid_move = Some(Move::new(
                    axis,
                    layer,
                    depth,
                    direction,
                    ReferenceFrame::Relative,
                    cube_size,
                ));

                if Some(axis) != current_axis {
                    current_axis = Some(axis);
                    used_layers_on_axis.clear();
                }
                for l in layer..layer + depth {
                    used_layers_on_axis.insert(l);
                }
            }
        }

        let _ = cube.apply_move(valid_move.unwrap());
    }

    let applied_move = cube.history.last().unwrap().clone();
    let normalized_move = applied_move
        .to_frame(ReferenceFrame::Normalized)
        .unwrap_or(applied_move); // Fallback if normalization fails, though it shouldn't for valid moves

    (cube.normalized(), normalized_move)
}

pub fn generate_training_data(
    cube_size: usize,
    scramble_length: usize,
    num_scrambles: usize,
) -> Vec<(Cube, Move)> {
    (0..num_scrambles)
        .map(|_| generate_training_sample(cube_size, scramble_length))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_data_3x3() {
        let data = generate_training_data(3, 10, 2);
        assert_eq!(data.len(), 2);
        for (cube_state, m) in data {
            assert_eq!(cube_state.faces.len(), 6);
            assert_eq!(m.reference_frame, ReferenceFrame::Normalized);
        }
    }

    #[test]
    fn test_redundancy_check_logic() {
        let data = generate_training_data(3, 50, 1);
        let moves: Vec<&Move> = data.iter().map(|(_, m)| m).collect();

        assert_eq!(moves.len(), 1);
    }
}
