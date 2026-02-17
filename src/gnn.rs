use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};

#[derive(Clone, Copy, Debug)]
pub enum NormalizationAlg {
    Symmetric,
    RandomWalk,
}

#[derive(Module, Debug)]
pub struct GCNConv<B: Backend> {
    linear: Linear<B>,
}

#[derive(Config, Debug)]
pub struct GCNConvConfig {
    pub d_input: usize,
    pub d_output: usize,
}

impl GCNConvConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GCNConv<B> {
        let linear = LinearConfig::new(self.d_input, self.d_output)
            .with_bias(false)
            .init(device);
        GCNConv { linear }
    }
}

impl<B: Backend> GCNConv<B> {
    pub fn forward<const D: usize>(&self, nodes: Tensor<B, D>, adj: Tensor<B, D>) -> Tensor<B, D> {
        let support = self.linear.forward(nodes);
        adj.matmul(support)
    }
}

pub fn normalize_adj<B: Backend>(adj: Tensor<B, 2>, alg: NormalizationAlg) -> Tensor<B, 2> {
    let size = adj.dims()[0];
    let device = adj.device();

    let adj = adj + Tensor::eye(size, &device);
    let deg = adj.clone().sum_dim(1);

    match alg {
        NormalizationAlg::Symmetric => {
            let deg_inv_sqrt = deg.powf_scalar(-0.5);
            adj * deg_inv_sqrt.clone() * deg_inv_sqrt.transpose()
        }
        NormalizationAlg::RandomWalk => {
            let deg_inv = deg.powf_scalar(-1.0);
            adj * deg_inv
        }
    }
}
