using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using System;
using System.Collections.Generic;
using System.Linq;
using ML_ProjectWork.ML;
using ML_ProjectWork.ML.Enum;

namespace ML_ProjectWork.Helpers
{
    /// <summary>
    ///     Вычисляет значимость фич
    /// </summary>
    internal static class FeatureHelper
    {
        #region Public methods

        /// <summary>
        ///     Тренирует модель, находит имена самых значимых фичей
        /// </summary>
        public static List<string> FindBestFeatures(IModel model, IDataView dataView)
        {
            var result = new List<string>();

            // подготовка pipeline
            var pipeline = model.MlContext.Transforms.Concatenate("Features", model.Features.ToArray())
                .Append(model.MlContext.Transforms.NormalizeLogMeanVariance("Features"));

            var trainer = model.SetTrainer();
            var trainedPipeline = pipeline.Append(trainer);

            // Предварительная обработка данных
            var preModel = trainedPipeline.Fit(dataView);
            var preprocessedData = preModel.Transform(dataView);

            var indexes = GetIndexes(model, trainer, preprocessedData);

            foreach (var index in indexes)
            {
                // Добавляю имена фич в порядке значимости
                result.Add(model.Features[index]);
            }

            return result;
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Возвращает индексы фич в порядке убывания значимости
        /// </summary>
        private static List<int> GetIndexes(IModel model, IEstimator<ITransformer> trainer, IDataView preprocessedData)
        {
            // Не получилось сделать универсально, так как тренеры вроду бы независимы
            if (model.Trainer == TrainerModel.FastForest)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<FastForestRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.LbfgsPoissonRegression)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<PoissonRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.Sdca)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<LinearRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.FastTree)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<FastTreeRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.FastTreeTweedie)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<FastTreeTweedieModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.Gam)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<GamRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.Ols)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<OlsModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.OnlineGradientDescent)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<LinearRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            if (model.Trainer == TrainerModel.LightGbm)
            {
                return FeaturePermutation(
                    model.MlContext,
                    (RegressionPredictionTransformer<LightGbmRegressionModelParameters>)trainer.Fit(preprocessedData),
                    preprocessedData,
                    model.Features.Count);
            }

            return new List<int>();
        }

        /// <summary>
        ///     Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        /// <param name="permutationCount"> Кол-во фич для расчета </param>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LinearModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Sdca & OnlineGradientDescent: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LinearRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     LightGbm: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LightGbmRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastForest: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<FastForestRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     LbfgsPoissonRegression: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<PoissonRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastTree: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<FastTreeRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastTreeTweedie: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<FastTreeTweedieModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Gam: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<GamRegressionModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Ols: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<OlsModelParameters> model,
            IDataView preprocessedTrainData,
            int permutationCount)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: permutationCount);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        #endregion
    }
}
