using Microsoft.ML.Data;

namespace ML_ProjectWork.Models
{
    /// <summary>
    ///     Модель дома
    /// </summary>
    public class HouseModel
    {
        #region Properties

        /// <summary>
        ///     Id записи
        /// </summary>
        [LoadColumn(0)]
        public float Id { get; set; }

        ///// <summary>
        /////     Дата создания записи
        ///// </summary>
        //[LoadColumn(1)]
        //public string Date { get; set; }

        /// <summary>
        ///     Цена дома
        /// </summary>
        [LoadColumn(2)]
        [ColumnName("Label")]
        public float Price { get; set; }

        /// <summary>
        ///     Кол-во спален
        /// </summary>
        [LoadColumn(3)]
        public float Bedrooms { get; set; }

        /// <summary>
        ///     Кол-во ванных комнат
        /// </summary>
        [LoadColumn(4)]
        public float Bathrooms { get; set; }

        /// <summary>
        ///     Размер жилой площади дома
        /// </summary>
        [LoadColumn(5)]
        public float LivingArea { get; set; }

        /// <summary>
        ///     Размер полной площади дома
        /// </summary>
        [LoadColumn(6)]
        public float Area { get; set; }

        /// <summary>
        ///     Кол-во этажей в доме
        /// </summary>
        [LoadColumn(7)]
        public float Floors { get; set; }

        /// <summary>
        ///     Есть ли выход к водоему
        /// </summary>
        [LoadColumn(8)]
        public float IsWaterFront { get; set; }

        /// <summary>
        ///     Оценка вида от 1 до 4
        /// </summary>
        [LoadColumn(9)]
        public float View { get; set; }

        /// <summary>
        ///     Условия проживания от 1 до 5
        /// </summary>
        [LoadColumn(10)]
        public float Condition { get; set; }

        /// <summary>
        ///     Оценка
        /// </summary>
        [LoadColumn(11)]
        public float Grade { get; set; }

        /// <summary>
        ///     Квадратные метры дома без подвала
        /// </summary>
        [LoadColumn(12)]
        public float SqftAbove { get; set; }

        /// <summary>
        ///     Квадратные метры дома с подвалом
        /// </summary>
        [LoadColumn(13)]
        public float SqftBasement { get; set; }

        /// <summary>
        ///     Год построения дома
        /// </summary>
        [LoadColumn(14)]
        public float YearBuilt { get; set; }

        /// <summary>
        ///     Год Обновления дома
        /// </summary>
        [LoadColumn(15)]
        public float YearRenovation { get; set; }

        /// <summary>
        ///     ZipCode
        /// </summary>
        [LoadColumn(16)]
        public float ZipCode { get; set; }

        /// <summary>
        ///     Широта
        /// </summary>
        [LoadColumn(17)]
        public float Lat { get; set; }

        /// <summary>
        ///     Долгота
        /// </summary>
        [LoadColumn(18)]
        public float Long { get; set; }

        /// <summary>
        ///     Средняя площадь дома из 15 ближайших домов
        /// </summary>
        [LoadColumn(19)]
        public float SqftLiving15 { get; set; }

        /// <summary>
        ///     Средняя площадь участка из 15 ближайших домов
        /// </summary>
        [LoadColumn(20)]
        public float SqftLot15 { get; set; }

        #endregion
    }
}
