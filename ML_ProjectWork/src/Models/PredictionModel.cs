﻿using ML_ProjectWork.ML;
using System;

namespace ML_ProjectWork.Models
{
    /// <summary>
    ///     Отвечает за предсказания
    /// </summary>
    internal class PredictionModel
    {
        #region Properties

        /// <summary>
        ///     Предсказание цены дома
        /// </summary>
        public float Score { get; set; }

        #endregion

        #region Public methods

        /// <summary>
        ///     Предсказывает цену дома, запускать только на нормализованных данных
        /// </summary>
        public static void Predict(RegressionModel model, HouseModel house, string expectedValue)
        {
            if (model.Model is null)
            {
                return;
            }

            var predEngine = model.MlContext.Model.CreatePredictionEngine<HouseModel, PredictionModel>(model.Model);

            var prediction = predEngine.Predict(house);

            Console.WriteLine($"Predicted price = {prediction.Score} | Expected value = {expectedValue}" +
                              $" for trainer: {model.Trainer}  (error = {model.MeanAbsoluteError} rsquared = {model.RSquared:P2})");
        }

        #endregion
    }
}
