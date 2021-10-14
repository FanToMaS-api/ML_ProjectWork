using System;
using System.Collections.Generic;
using System.Reflection;

namespace ML_ProjectWork.Helpers
{
    /// <summary>
    ///     Нормализует объект
    /// </summary>
    /// <remarks>
    ///     Используется нормализация средним (Z-нормализация)
    ///     https://wiki.loginom.ru/articles/data-normalization.html
    /// </remarks>
    internal class Normalizer<T>
    where T : class
    {
        #region Fields

        /// <summary>
        ///     Словарь значений диспресии для каждой из фич
        /// </summary>
        private readonly Dictionary<string, float> _dispersions;

        /// <summary>
        ///     Словарь средних значений для каждой из фич
        /// </summary>
        private readonly Dictionary<string, float> _meanValues;

        /// <summary>
        ///     Массив объектов, на основе которых получаем данные по нормализации
        /// </summary>
        private readonly T[] _models;

        /// <summary>
        ///     Массив фич объекта
        /// </summary>
        private readonly PropertyInfo[] _features;

        #endregion

        #region .ctor

        /// <summary>
        ///     Вычисляет значения дисперсии и среднего значения для каждой из фич
        /// </summary>
        public Normalizer(T[] models)
        {
            _models = models;
            _features = models[0].GetType().GetProperties();
            _meanValues = GetMeanValues();
            _dispersions = GetDispersions();
        }

        #endregion

        #region Public methods

        /// <summary>
        ///     Нормализует объект на основе массива обектов
        /// </summary>
        public void Normalize(T obj)
        {
            foreach (var feature in _features)
            {
                var currentValue = Convert.ToSingle(feature.GetValue(obj));
                var newValue = (currentValue - Convert.ToSingle(_meanValues[feature.Name])) /
                               Math.Sqrt(Convert.ToSingle(_dispersions[feature.Name]));

                feature.SetValue(obj, (float)newValue);
            }
        }

        #endregion

        #region Private methods

        /// <summary>
        ///     Вычисляет дисперсии для каждой из фич
        /// </summary>
        private Dictionary<string, float> GetDispersions()
        {
            var res = new Dictionary<string, float>();
            foreach (var feature in _features)
            {
                float dispersion = 0;
                foreach (var model in _models)
                {
                    var deviation = Convert.ToSingle(feature.GetValue(model)) - Convert.ToSingle(_meanValues[feature.Name]);
                    dispersion += (float)Math.Pow(deviation, 2);
                }

                dispersion /= _models.Length;
                res.Add(feature.Name, dispersion);
            }

            return res;
        }

        /// <summary>
        ///     Вычисляет средние значения для каждой из фич
        /// </summary>
        private Dictionary<string, float> GetMeanValues()
        {
            var res = new Dictionary<string, float>();
            foreach (var feature in _features)
            {
                var meanValue = (float)0.0;
                foreach (var model in _models)
                {
                    meanValue += Convert.ToSingle(feature.GetValue(model));
                }

                meanValue /= _models.Length;
                res.Add(feature.Name, meanValue);
            }

            return res;
        }

        #endregion
    }
}
