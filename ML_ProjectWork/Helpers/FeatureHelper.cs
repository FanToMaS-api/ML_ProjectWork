using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace ML_ProjectWork
{
    /// <summary>
    ///     Вычисляет значимость фич
    /// </summary>
    internal static class FeatureHelper
    {
        /// <summary>
        ///     Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(
            MLContext mlColntext,
            RegressionPredictionTransformer<LinearModelParameters> model,
            IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

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
            IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

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
            IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

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
            IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     LbfgsPoissonRegression: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(MLContext mlColntext, RegressionPredictionTransformer<PoissonRegressionModelParameters> model, IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastTree: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(MLContext mlColntext, RegressionPredictionTransformer<FastTreeRegressionModelParameters> model, IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     FastTreeTweedie: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(MLContext mlColntext, RegressionPredictionTransformer<FastTreeTweedieModelParameters> model, IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Gam: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(MLContext mlColntext, RegressionPredictionTransformer<GamRegressionModelParameters> model, IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }

        /// <summary>
        ///     Ols: Выводит фичи в порядке уменьшения влияния на результат
        /// </summary>
        public static List<int> FeaturePermutation(MLContext mlColntext, RegressionPredictionTransformer<OlsModelParameters> model, IDataView preprocessedTrainData)
        {
            var featuresPermutation = mlColntext.Regression.PermutationFeatureImportance(
                model,
                preprocessedTrainData,
                permutationCount: 5);

            return featuresPermutation
                .Select((metric, index) => new { index, metric.RSquared })
                .OrderByDescending(_ => Math.Abs(_.RSquared.Mean))
                .Select(_ => _.index).ToList();
        }
    }
}
