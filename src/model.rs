use crate::cube::get_adjacency_matrix;
use crate::gnn::{GCNConv, GCNConvConfig, NormalizationAlg, normalize_adj};
use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    tensor::{Tensor, backend::Backend},
};

#[derive(Clone, Debug)]
pub struct Prediction<B: Backend> {
    // Policy Heads
    pub axis: Tensor<B, 2>,
    pub slice: Tensor<B, 2>,
    pub depth: Tensor<B, 2>,
    pub direction: Tensor<B, 2>,

    // Value Head
    pub value: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct CubeGNN<B: Backend> {
    embedding: GCNConv<B>,
    hidden: Vec<GCNConv<B>>,

    // Policy Heads
    axis: Linear<B>,
    slice: Linear<B>,
    depth: Linear<B>,
    direction: Linear<B>,

    // Value Head
    value: Linear<B>,

    adj: Tensor<B, 3>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct CubeGNNConfig {
    pub cube_size: usize,
    pub num_layers: usize,
    pub d_hidden: usize,
}

impl CubeGNNConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CubeGNN<B> {
        let adj_raw = get_adjacency_matrix::<B>(self.cube_size, device);
        let adj = normalize_adj(adj_raw, NormalizationAlg::Symmetric);
        let adj = adj.unsqueeze_dim(0);

        let d_input = 6; // One-hot colors
        let head_input_dim = self.d_hidden * 6 * self.cube_size.pow(2); // After flattening the hidden states

        let embedding = GCNConvConfig::new(d_input, self.d_hidden).init(device);
        let hidden =
            vec![GCNConvConfig::new(self.d_hidden, self.d_hidden).init(device); self.num_layers];

        let axis = LinearConfig::new(head_input_dim, 3).init(device);
        let slice = LinearConfig::new(head_input_dim, self.cube_size).init(device);
        let depth = LinearConfig::new(head_input_dim, self.cube_size / 2).init(device);
        let direction = LinearConfig::new(head_input_dim, 3).init(device);
        let value = LinearConfig::new(head_input_dim, 1).init(device);

        CubeGNN {
            embedding,
            hidden,
            axis,
            slice,
            depth,
            direction,
            value,
            adj,
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> CubeGNN<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Prediction<B> {
        let mut x = self.embedding.forward(input, self.adj.clone());
        x = self.activation.forward(x);

        for layer in &self.hidden {
            x = layer.forward(x, self.adj.clone());
            x = self.activation.forward(x);
        }

        let x = x.flatten(1, 2);

        Prediction {
            axis: self.axis.forward(x.clone()),
            slice: self.slice.forward(x.clone()),
            depth: self.depth.forward(x.clone()),
            direction: self.direction.forward(x.clone()),
            value: self.value.forward(x),
        }
    }
}
