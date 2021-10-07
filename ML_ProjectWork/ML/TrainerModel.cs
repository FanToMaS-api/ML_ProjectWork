namespace ML_ProjectWork
{
    /// <summary>
    ///     Определяет модели тренеров
    /// </summary>
    internal enum TrainerModel
    {
        /// <summary>
        ///     LbfgsPoissonRegression
        /// </summary>
        LbfgsPoissonRegression,

        /// <summary>
        ///     FastForest
        /// </summary>
        FastForest,

        /// <summary>
        ///     FastTree
        /// </summary>
        FastTree,

        /// <summary>
        ///     FastTreeTweedie
        /// </summary>
        FastTreeTweedie,

        /// <summary>
        ///     Gam
        /// </summary>
        Gam,

        /// <summary>
        ///     LightGbm
        /// </summary>
        LightGbm,

        /// <summary>
        ///     Ols
        /// </summary>
        Ols,

        /// <summary>
        ///     OnlineGradientDescent
        /// </summary>
        OnlineGradientDescent,

        /// <summary>
        ///     Sdca
        /// </summary>
        Sdca
    }
}
