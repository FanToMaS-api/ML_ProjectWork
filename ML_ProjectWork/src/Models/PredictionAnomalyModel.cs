using Microsoft.ML.Data;

namespace ML_ProjectWork.Models
{
    /// <summary>
    ///     Модель для хранения данных об аномалиях
    /// </summary>
    internal class PredictionAnomalyModel
    {
        // vector to hold anomaly detection results. Including isAnomaly, anomalyScore, magnitude, expectedValue, boundaryUnits, upperBoundary and lowerBoundary.
        [VectorType(7)]
        public double[] Preds { get; set; }
    }
}
