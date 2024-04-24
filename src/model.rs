use burn::{
    config::Config,
    module::Module,
    nn::{loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::TitanicBatch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    hidden_layer: Linear<B>,
    second_hidden_layer: Linear<B>,
    output_layer: Linear<B>,
    dropout: Dropout,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();

        let x = self.input_layer.forward(x);
        let x = self.dropout.forward(x);
        // let x = self.activation.forward(x);
        let x = self.hidden_layer.forward(x);
        let x = self.dropout.forward(x);
        // let x = self.activation.forward(x);
        let x = self.second_hidden_layer.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: TitanicBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets.unsqueeze();
        let output = self.forward(item.inputs);

        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_features: usize,
    #[config(default = 48)]
    hidden_size: usize,
    #[config(default = "0.35")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input_layer: LinearConfig::new(self.num_features, self.num_features * 2)
                .with_bias(true)
                .init(device),
            hidden_layer: LinearConfig::new(self.num_features * 2, self.hidden_size)
                .with_bias(true)
                .init(device),
            second_hidden_layer: LinearConfig::new(self.hidden_size, self.hidden_size / 2)
                .with_bias(true)
                .init(device),
            output_layer: LinearConfig::new(self.hidden_size / 2, 2)
                .with_bias(true)
                .init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<TitanicBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: TitanicBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TitanicBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: TitanicBatch<B>) -> ClassificationOutput<B> {
        self.forward_step(item)
    }
}
