from pathlib import Path
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger
from src.core.singleton import SingletonMeta
import torch
from entity.config_entity import BERTConfig, CNNClassifierConfig, CallbacksConfig, DatasetConfig, EDAConfig, EarlyStoppingCallbackConfig, GradientClipCallbackConfig, LOGREGConfig, LRSchedulerCallbackConfig, LSTMATTENTIONConfig, LSTMClassifierConfig, ModelCheckpointCallbackConfig, ModelConfig, ModelEvaluationConfig, ModelTrainerConfig, PreprocessingConfig, SBERTConfig, SVMConfig
from sentiment_analysis.utils.helpers import get_num_workers
import os

class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        logger.info("Getting text preprocessing config")
        config = self.config.data
        logger.info(f"Text preprocessing config params: {config}")

        preprocessing_config = PreprocessingConfig(
            dataset_name=config.dataset_name,
            batch_size=config.batch_size,
            max_length=config.max_length,
            test_split_ratio=config.test_split_ratio,
            seed=config.seed,
            tokenizer_name=config.tokenizer_name
        )
        logger.info(f"PreprocessingConfig config created: {preprocessing_config}")
        return preprocessing_config
    
    def get_dataset_config(self) -> DatasetConfig:
        """
        Retrieves DataLoader specific configuration parameters.
        """
        logger.info("Getting Dataset and DataLoader configuration.")
        config = self.config.data
        batch_size = config.batch_size
        num_workers = get_num_workers(is_training=True)
        pin_memory = torch.cuda.is_available() # True if CUDA is available

        dataset_config = DatasetConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        logger.info(f"DatasetConfig created: {dataset_config}")
        return dataset_config
    



    def get_eda_config(self) -> EDAConfig:
        """
        Retrieves Exploratory Data Analysis specific configuration parameters.
        """
        logger.info("Getting EDA configuration.")
        config = self.config.eda 
        params = self.params.eda

        dirs_to_create = [config.report_dir]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for EDA reports: {dirs_to_create}")
        
        eda_config = EDAConfig(
            report_dir=Path(config.report_dir), 
            max_words_to_plot=params.get('max_words_to_plot', 20),
            min_word_len=params.get('min_word_len', 3)
        )
        logger.info(f"EDAConfig created: {eda_config}")
        return eda_config
    

    def get_lstm_config(self, model_type: str) -> LSTMClassifierConfig:
        """
        Retrieves LSTM model configuration parameters.
        """
        logger.info("Getting LSTM configuration.")
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )

        lstm_config = LSTMClassifierConfig(
            embedding_dim=params.embedding_dim,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            dropout=params.dropout,
            bidirectional=params.bidirectional,
            batch_first=params.batch_first
        )

        logger.info(f"LSTMClassifierConfig created: {lstm_config}")
        return lstm_config
    
    def get_cnn_config(self, model_type: str) -> CNNClassifierConfig:
        """
        Retrieves CNN model configuration parameters.
        """
        logger.info("Getting CNN configuration.")
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )

        cnn_config = CNNClassifierConfig(
            num_filters=params.num_filters,
            filter_sizes=params.filter_sizes,
            dropout=params.dropout,
            embedding_dim=params.embedding_dim
        )

        logger.info(f"CNNClassifierConfig created: {cnn_config}")
        return cnn_config

    def get_lstm_attention_config(self, model_type: str) -> LSTMATTENTIONConfig:
        """
        Retrieves LSTM Attention model configuration parameters.
        """
        logger.info("Getting LSTM Attention configuration.")
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )

        lstm_attention_config = LSTMATTENTIONConfig(
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            dropout=params.dropout,
            bidirectional=params.bidirectional,
            attention_size=params.attention_size
        )

        logger.info(f"LSTMATTENTIONConfig created: {lstm_attention_config}")
        return lstm_attention_config

    def get_bert_config(self, model_type: str) -> BERTConfig:
        """
        Retrieves BERT model configuration parameters.
        """
        logger.info("Getting BERT configuration.")
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )

        bert_config = BERTConfig(
            pretrained_model_name=params.pretrained_model_name,
            dropout=params.dropout,
            fine_tune=params.fine_tune
        )

        logger.info(f"BERTConfig created: {bert_config}")
        return bert_config

    def get_sbert_config(self, model_type: str) -> SBERTConfig:
        """
        Retrieves SBERT model configuration parameters.
        """
        logger.info("Getting SBERT configuration.")
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )

        sbert_config = SBERTConfig(
            pretrained_model_name=params.pretrained_model_name,
            dropout=params.dropout,
            fine_tune=params.fine_tune
        )

        logger.info(f"SBERTConfig created: {sbert_config}")
        return sbert_config

    def get_logreg_config(self, model_type: str) -> LOGREGConfig:
        """
        Retrieves Logistic Regression model configuration parameters.
        """
        logger.info("Getting Logistic Regression configuration.")
        
    def get_svm_config(self, model_type: str) -> SVMConfig:
        """
        Retrieves SVM model configuration parameters.
        """
        logger.info("Getting SVM configuration.")
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )

        svm_config = SVMConfig(
            input_size=params.input_size,
            num_classes=params.num_classes,
            kernel=params.kernel,
            C=params.C
        )

        logger.info(f"SVMConfig created: {svm_config}")
        return svm_config
    def get_model_config(self) -> ModelConfig:
        """
        Retrieves model configuration parameters.
        """
        logger.info("Getting Model configuration.")
        # Get base config
        config = self.config.model
        model_type = config.model_type.upper()
        params = self.params.model_options.get(model_type, {})

         # Validate model exists
        if not params:
            available_models = list(self.params.model_options.keys())
            raise ValueError(
                f"❌ Model '{model_type}' not found in model_options. "
                f"Available: {available_models}"
            )
        
        model_config = ModelConfig(
            model_type=model_type,
            LSTM=self.get_lstm_config(model_type),
            CNN=self.get_cnn_config(model_type),
            LSTMATTENTION=self.get_lstm_attention_config(model_type),
            BERT=self.get_bert_config(model_type),
            SBERT=self.get_sbert_config(model_type),
            LOGREG=self.get_logreg_config(model_type),
            SVM=self.get_svm_config(model_type)
        )
        logger.info(f"ModelConfig created: {model_config}")
        return model_config

    
    def get_early_stopping_config(self) -> EarlyStoppingCallbackConfig:
        """
        Retrieves Early Stopping callback configuration parameters.
        """
        logger.info("Getting Early Stopping Callback configuration.")
        params = self.params.callbacks.early_stopping
        early_stopping_config = EarlyStoppingCallbackConfig(
            patience=params.patience,
            mode=params.mode,
            min_delta=params.min_delta
        )
        logger.info(f"EarlyStoppingCallbackConfig created: {early_stopping_config}")
        return early_stopping_config
    

    def get_lr_scheduler_config(self) -> LRSchedulerCallbackConfig:
        """
        Retrieves Learning Rate Scheduler callback configuration parameters.
        """
        logger.info("Getting LR Scheduler Callback configuration.")
        params = self.params.callbacks.lr_scheduler
        lr_scheduler_config = LRSchedulerCallbackConfig(
            patience=params.patience,
            factor=params.factor,
            mode=params.mode
        )
        logger.info(f"LRSchedulerCallbackConfig created: {lr_scheduler_config}")
        return lr_scheduler_config
    
    def get_gradient_clip_config(self) -> GradientClipCallbackConfig:
        """
        Retrieves Gradient Clipping callback configuration parameters.
        """
        logger.info("Getting Gradient Clip Callback configuration.")
        config = self.config.callbacks.gradient_clip
        gradient_clip_config = GradientClipCallbackConfig(
            max_norm=config.max_norm
        )
        logger.info(f"GradientClipCallbackConfig created: {gradient_clip_config}")
        return gradient_clip_config
    

    def get_model_checkpoint_config(self) -> ModelCheckpointCallbackConfig:
        """
        Retrieves Model Checkpoint callback configuration parameters.
        """
        logger.info("Getting Model Checkpoint Callback configuration.")
        config = self.config.callbacks.model_checkpoint

        dirs_to_create = [config.checkpoint_dir]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for Model Checkpoints: {dirs_to_create}")

        model_checkpoint_config = ModelCheckpointCallbackConfig(
            checkpoint_dir=Path(config.checkpoint_dir)
        )
        logger.info(f"ModelCheckpointCallbackConfig created: {model_checkpoint_config}")
        return model_checkpoint_config
    

    def get_callbacks_config(self) -> CallbacksConfig:
        """
        Retrieves overall Callbacks configuration parameters.
        """
        logger.info("Getting Callbacks configuration.")
        callbacks_config = CallbacksConfig(
            early_stopping_callback_config=self.get_early_stopping_config(),
            lr_scheduler_callback_config=self.get_lr_scheduler_config(),
            model_checkpoint_callback_config=self.get_model_checkpoint_config(),
            gradient_clip_callback_config=self.get_gradient_clip_config()
        )
        logger.info(f"CallbacksConfig created: {callbacks_config}")
        return callbacks_config

    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Retrieves Model Trainer configuration parameters.
        """
        logger.info("Getting Model Trainer configuration.")
        params = self.params.trainer
        config = self.config.trainer

        dirs_to_create = [config.report_dir]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for Trainer reports: {dirs_to_create}")
        
        model_trainer_config = ModelTrainerConfig(
            max_epochs=params.max_epochs,
            learning_rate=params.learning_rate,
            gradient_clip_norm=params.gradient_clip_norm,
            report_dir=Path(config.report_dir)
        )
        logger.info(f"ModelTrainerConfig created: {model_trainer_config}")
        return model_trainer_config

    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieves Model Evaluation configuration parameters.
        """
        logger.info("Getting Model Evaluation configuration.")
        config = self.config.evaluation

        dirs_to_create = [config.report_dir]
        create_directory(dirs_to_create)
        logger.info(f"Created directories for Evaluation reports: {dirs_to_create}")
        if os.path.exists(config.model_path):
            logger.info(f"Model path exists: {config.model_path}")
        else:
            raise ValueError(f"❌ Model path does not exist: {config.model_path}")
        
        model_evaluation_config = ModelEvaluationConfig(
            model_path=Path(config.model_path),
            report_dir=Path(config.report_dir)
        )
        logger.info(f"ModelEvaluationConfig created: {model_evaluation_config}")
        return model_evaluation_config