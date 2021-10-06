using System.Linq;
using Microsoft.ML;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Занимается созданием и обучением модели
    /// </summary>
    internal class Model
    {
        #region Fields

        private readonly string _dataPath;

        private readonly TrainerModel _trainer;

        private readonly double _testFraction;

        private readonly char _separatorChar;

        private readonly int _countExcludedFeatures;

        private string[] _featureNames;

        private IDataView? _dataView;

        private readonly MLContext _mlContext;

        #endregion

        #region .ctor

        /// <inheritdoc cref="Model"/>
        public Model(string dataPath, TrainerModel _trainer, int countExcludedFeatures, TrainerModel trainer, char separatorChar = ',', double testFraction = 0.2, int seed = 212103)
        {
            _mlContext = new MLContext(seed);
            _dataPath = dataPath;
            _countExcludedFeatures = countExcludedFeatures;
            this._trainer = trainer;

            _separatorChar = separatorChar;
            _testFraction = testFraction;
            _dataView = _mlContext.Data.LoadFromTextFile<HouseModel>(_dataPath, separatorChar: _separatorChar);
            _featureNames = _dataView.Schema.Select(_ => _.Name).ToArray();
        }

        #endregion

        #region Public methods

        /// <summary>
        ///     Проводит обучение модели
        /// </summary>
        public void Fit()
        {
            var excludedFeatures = FindBestFeatures();
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Тренирует модель, находит индексы самых значимых фичей
        /// </summary>
        public void FindBestFeatures()
        {
            var pipeline = _mlContext.Transforms.Concatenate("Features", _featureNames)
                .Append(_mlContext.Transforms.NormalizeLogMeanVariance("Features"));

            var trainer = _mlContext.Regression.Trainers.LightGbm();
            var trainedPipeline = pipeline.Append(trainer);

            var model = trainedPipeline.Fit(_dataView);
            var preprocessedData = model.Transform(_dataView);

            FeatureHelper.FeaturePermutation(_mlContext, trainer.Fit(preprocessedData), preprocessedData, _featureNames);
        }

        /// <summary>
        ///     Готовит данные к обучению
        /// </summary>
        private void DataPreparing()
        {

        }

        private void SetTrainer()
        {
            if()
        }

        #endregion
    }
}
