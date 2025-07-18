use burn::{
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        conv::{Conv3d, Conv3dConfig},
    },
    prelude::*,
    tensor::activation::{relu, softmax},
};

#[derive(Module, Debug)]
pub struct RubiksModel<B: Backend> {
    embed: Embedding<B>,
    conv1: Conv3d<B>,
    dense0: Linear<B>,
    dense1: Linear<B>,
}

impl<B: Backend> RubiksModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(6, 3).init(device);
        let conv1 = Conv3dConfig::new([3, 32], [2, 2, 2]).init(device);
        let dense0 = LinearConfig::new(32 * 5 * 2 * 2, 128).init(device);
        let dense1 = LinearConfig::new(128, 18).init(device);

        Self {
            embed,
            conv1,
            dense0,
            dense1,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];
        let input = input.flatten(1, 3);
        let embedded = self.embed.forward(input.int());
        let emb_size = embedded.dims()[2];
        let embedded = embedded.reshape([batch_size, 6, 3, 3, emb_size]);

        let embedded = embedded.permute([0, 4, 1, 2, 3]); // Permute so that embeddings are channels: [batch_size, emb_size, 6, 3, 3]
        let conv_out = relu(self.conv1.forward(embedded));

        let flattened = conv_out.flatten(1, 4);
        let dense_out0 = relu(self.dense0.forward(flattened));
        let dense_out1 = self.dense1.forward(dense_out0);
        softmax(dense_out1, 1)
    }
}
