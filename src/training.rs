use crate::{data::TitanicBatcher, dataset::TitanicDataset, model::ModelConfig};
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::CudaMetric;
use burn::train::MetricEarlyStoppingStrategy;
use burn::train::StoppingCondition;
use burn::{config::Config, data::dataset::Dataset, record::NoStdTrainingRecorder};
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

const ARTIFACT_DIR: &'static str = "/tmp/titanic";

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimzer: AdamConfig,
    #[config(default = 1500)]
    pub num_epochs: usize,
    #[config(default = 512)]
    pub batch_size: usize,
    #[config(default = 24)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let optimizer = AdamConfig::new();
    let config = TrainingConfig::new(ModelConfig::new(), optimizer).with_num_epochs(5000);
    B::seed(config.seed);

    let train_dataset = TitanicDataset::train();
    let test_dataset = TitanicDataset::test();

    println!("Train data is {} entries", train_dataset.len());
    println!("Test data is {} entries", test_dataset.len());

    let batcher_train = TitanicBatcher::<B>::new(device.clone());
    let batcher_test = TitanicBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    // Model
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        // .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
        //     Aggregate::Mean,
        //     Direction::Lowest,
        //     Split::Valid,
        //     StoppingCondition::NoImprovementSince { n_epochs: 50 },
        // ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimzer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .unwrap();

    model_trained
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}
