use crate::cube::{
    POSSIBLE_MOVE_COUNT, UnvariantMove, compare_move_type, compare_unvariant, cube_to_tensor,
    move_to_number, move_to_opposite, move_to_unvariant, number_to_move, solve_cube,
};
use burn::{prelude::*, tensor::ElementConversion};
use cubesim::FaceletCube;
use cubesim::prelude::{Cube, Move};
use rand::prelude::IndexedRandom;

#[derive(Clone)]
struct Tree {
    end: bool,
    children: Box<[Option<Tree>; POSSIBLE_MOVE_COUNT]>,
    pending_moves: Vec<UnvariantMove>,
}
impl Tree {
    pub fn new() -> Self {
        Self {
            end: false,
            children: Default::default(),
            pending_moves: Vec::new(),
        }
    }
    pub fn add(&mut self, mv: Move, end: bool) -> &mut Tree {
        if self.end {
            panic!("Cannot add moves to a terminal node");
        }

        let unvar_op_mv = move_to_unvariant(move_to_opposite(mv));
        let mut pending_moves = self.pending_moves.clone();

        pending_moves
            .retain(|&m| compare_unvariant(m, unvar_op_mv));
        pending_moves.push(move_to_unvariant(mv));

        let mv_index = move_to_number(mv);
        self.children[mv_index] = Some(Self {
            end,
            children: Default::default(),
            pending_moves,
        });
        self.children[mv_index].as_mut().unwrap()
    }
    pub fn clear(&mut self) {
        self.children.fill(None);
    }
    pub fn available(&self) -> Vec<Move> {
        if self.end {
            return Vec::new();
        }

        let mut move_indices: [Option<usize>; POSSIBLE_MOVE_COUNT] =
            core::array::from_fn(|i| Some(i));
        for (index, child) in self.children.iter().enumerate() {
            if let Some(child) = child {
                if child.available().len() == 0 {
                    move_indices[index] = None;
                }
            }
        }
        move_indices
            .into_iter()
            .flatten()
            .map(number_to_move)
            .filter(|&mv| {
                self.pending_moves.iter().all(|&m| !compare_move_type(mv, m))
            })
            .collect()
    }
}

pub struct Batch<B: Backend> {
    pub states: Tensor<B, 4>,
    pub scrambles: Tensor<B, 1, Int>,
}

pub struct Data<'a, B: Backend> {
    device: &'a B::Device,
    tree: Tree,
    pub batch_size: usize,
    pub scramble_moves: usize,
}
impl<'a, B: Backend> Data<'a, B> {
    pub fn new(device: &'a B::Device, batch_size: usize, scramble_moves: usize) -> Self {
        Self {
            device,
            tree: Tree::new(),
            batch_size,
            scramble_moves,
        }
    }
}
impl<'a, B: Backend> Iterator for Data<'a, B> {
    type Item = Batch<B>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut states = Vec::new();
        let mut scrambles = Vec::new();
        let mut rng = rand::rng();
        let mut count = 0;
        while count < self.batch_size {
            let mut tree = &mut self.tree;
            if tree.available().is_empty() {
                tree.clear();
            }
            let mut scramble = Vec::new();
            for i in 0..self.scramble_moves {
                let available_moves = tree.available();
                let mv = available_moves.choose(&mut rng).unwrap();

                scramble.push(*mv);
                tree = tree.add(*mv, i == self.scramble_moves - 1);
            }
            let cube = FaceletCube::new(3).apply_moves(&scramble);
            let mut solution = solve_cube(&cube);

            if solution.len() > self.scramble_moves {
                // println!(
                //     "Warning: Solution length ({}) is greater than the number of moves ({}) used to scramble the cube",
                //     solution.len(),
                //     self.scramble_moves
                // );
                // println!("Suboptimal solution found, falling back to scramble");
                solution = scramble.clone();
            } else if solution.len() < self.scramble_moves {
                println!(
                    "Warning: Solution length ({}) is less than the number of moves ({}) used to scramble the cube",
                    solution.len(),
                    self.scramble_moves
                );
                println!("Retrying with a new scramble");
                continue; // Retry with a new scramble
            }

            let cube_tensor = cube_to_tensor::<B>(self.device, &cube);
            let scramble_tensor = Tensor::<B, 1, Int>::from_data(
                [(move_to_number(solution[0]) as i64).elem::<B::IntElem>()],
                self.device,
            );

            let cube_tensor = cube_tensor.unsqueeze::<4>();

            states.push(cube_tensor);
            scrambles.push(scramble_tensor);
            count += 1;
        }

        Some(Batch {
            states: Tensor::cat(states, 0),
            scrambles: Tensor::cat(scrambles, 0),
        })
    }
}
