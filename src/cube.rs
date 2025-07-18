use burn::prelude::*;
use cubesim::prelude::{Cube, Move, MoveVariant};
use cubesim::{Face, FaceletCube};
use itertools::Itertools;
use kociemba::moves::Move as KMove;

pub type UnvariantMove = fn(MoveVariant) -> Move;

pub const MOVES: [UnvariantMove; 6] = [Move::U, Move::L, Move::F, Move::R, Move::B, Move::D];
pub const MOVE_VARIANTS: [MoveVariant; 3] = [
    MoveVariant::Standard,
    MoveVariant::Inverse,
    MoveVariant::Double,
];
pub const POSSIBLE_MOVE_COUNT: usize = MOVES.len() * MOVE_VARIANTS.len();

fn data_to_array(data: Vec<Vec<Vec<f32>>>) -> Option<[[[f32; 3]; 3]; 6]> {
    if data.len() != 6
        || data
            .iter()
            .any(|m| m.len() != 3 || m.iter().any(|r| r.len() != 3))
    {
        return None;
    }

    let mut array = [[[0.0f32; 3]; 3]; 6];

    for (i, matrix) in data.into_iter().enumerate() {
        for (j, row) in matrix.into_iter().enumerate() {
            for (k, val) in row.into_iter().enumerate() {
                array[i][j][k] = val;
            }
        }
    }

    Some(array)
}

pub fn solve_cube(cube: &FaceletCube) -> Vec<Move> {
    let cube_str = stringify_cube(cube);
    let solution = kociemba::solver::solve(&cube_str, 20, 3.0).expect("Failed to solve the cube");
    solution.solution.into_iter().map(convert_move).collect()
}

pub fn cube_to_tensor<B: Backend>(device: &B::Device, cube: &FaceletCube) -> Tensor<B, 3> {
    let cube_data: [[[f32; 3]; 3]; 6] = data_to_array(
        cube.state()
            .into_iter()
            .map(|face| face_to_number(face) as f32)
            .chunks(3)
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .chunks(3)
            .into_iter()
            .map(|chunk| chunk.collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )
    .expect("Invalid cube state");
    Tensor::from_data(cube_data, device)
}

pub fn convert_move(mv: KMove) -> Move {
    match mv {
        KMove::U => Move::U(MoveVariant::Standard),
        KMove::U2 => Move::U(MoveVariant::Double),
        KMove::U3 => Move::U(MoveVariant::Inverse),
        KMove::L => Move::L(MoveVariant::Standard),
        KMove::L2 => Move::L(MoveVariant::Double),
        KMove::L3 => Move::L(MoveVariant::Inverse),
        KMove::F => Move::F(MoveVariant::Standard),
        KMove::F2 => Move::F(MoveVariant::Double),
        KMove::F3 => Move::F(MoveVariant::Inverse),
        KMove::R => Move::R(MoveVariant::Standard),
        KMove::R2 => Move::R(MoveVariant::Double),
        KMove::R3 => Move::R(MoveVariant::Inverse),
        KMove::B => Move::B(MoveVariant::Standard),
        KMove::B2 => Move::B(MoveVariant::Double),
        KMove::B3 => Move::B(MoveVariant::Inverse),
        KMove::D => Move::D(MoveVariant::Standard),
        KMove::D2 => Move::D(MoveVariant::Double),
        KMove::D3 => Move::D(MoveVariant::Inverse),
    }
}

pub fn compare_move_type(mv1: Move, mv2: UnvariantMove) -> bool {
    let move_variant = mv1.get_variant();
    let mv2 = mv2(move_variant);
    mv1 == mv2
}

pub fn compare_unvariant(mv1: UnvariantMove, mv2: UnvariantMove) -> bool {
    let mv1 = mv1(MoveVariant::Standard);
    let mv2 = mv2(MoveVariant::Standard);
    mv1 == mv2
}

pub fn move_to_unvariant(mv: Move) -> UnvariantMove {
    match mv {
        Move::U(_) => Move::U,
        Move::L(_) => Move::L,
        Move::F(_) => Move::F,
        Move::R(_) => Move::R,
        Move::B(_) => Move::B,
        Move::D(_) => Move::D,
        _ => panic!("Invalid move"),
    }
}

pub fn move_to_opposite(mv: Move) -> Move {
    match mv {
        Move::U(variant) => Move::D(variant),
        Move::L(variant) => Move::R(variant),
        Move::F(variant) => Move::B(variant),
        Move::R(variant) => Move::L(variant),
        Move::B(variant) => Move::F(variant),
        Move::D(variant) => Move::U(variant),
        _ => panic!("Invalid move"),
    }
}

pub fn move_to_number(mv: Move) -> usize {
    match mv {
        Move::U(MoveVariant::Standard) => 0,
        Move::U(MoveVariant::Double) => 1,
        Move::U(MoveVariant::Inverse) => 2,
        Move::L(MoveVariant::Standard) => 3,
        Move::L(MoveVariant::Double) => 4,
        Move::L(MoveVariant::Inverse) => 5,
        Move::F(MoveVariant::Standard) => 6,
        Move::F(MoveVariant::Double) => 7,
        Move::F(MoveVariant::Inverse) => 8,
        Move::R(MoveVariant::Standard) => 9,
        Move::R(MoveVariant::Double) => 10,
        Move::R(MoveVariant::Inverse) => 11,
        Move::B(MoveVariant::Standard) => 12,
        Move::B(MoveVariant::Double) => 13,
        Move::B(MoveVariant::Inverse) => 14,
        Move::D(MoveVariant::Standard) => 15,
        Move::D(MoveVariant::Double) => 16,
        Move::D(MoveVariant::Inverse) => 17,
        _ => panic!("Invalid move"),
    }
}

pub fn number_to_move(num: usize) -> Move {
    match num {
        0 => Move::U(MoveVariant::Standard),
        1 => Move::U(MoveVariant::Double),
        2 => Move::U(MoveVariant::Inverse),
        3 => Move::L(MoveVariant::Standard),
        4 => Move::L(MoveVariant::Double),
        5 => Move::L(MoveVariant::Inverse),
        6 => Move::F(MoveVariant::Standard),
        7 => Move::F(MoveVariant::Double),
        8 => Move::F(MoveVariant::Inverse),
        9 => Move::R(MoveVariant::Standard),
        10 => Move::R(MoveVariant::Double),
        11 => Move::R(MoveVariant::Inverse),
        12 => Move::B(MoveVariant::Standard),
        13 => Move::B(MoveVariant::Double),
        14 => Move::B(MoveVariant::Inverse),
        15 => Move::D(MoveVariant::Standard),
        16 => Move::D(MoveVariant::Double),
        17 => Move::D(MoveVariant::Inverse),
        _ => panic!("Invalid move number"),
    }
}

pub fn face_to_number(face: Face) -> usize {
    match face {
        Face::U => 0,
        Face::L => 1,
        Face::F => 2,
        Face::R => 3,
        Face::B => 4,
        Face::D => 5,
        _ => panic!("Invalid facelet character"),
    }
}

pub fn stringify_cube(cube: &FaceletCube) -> String {
    cube.state()
        .into_iter()
        .map(|face| face.to_string())
        .join("")
}

pub fn encode_cube(stickers: &Vec<Face>) -> u128 {
    let mut value: u128 = 0;
    for &sticker in stickers.iter().take(42) {
        value <<= 3;
        value |= face_to_number(sticker) as u128;
    }
    value
}
