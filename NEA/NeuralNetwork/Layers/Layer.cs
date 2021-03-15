namespace NEA.NeuralNetwork.Layers
{
    using Tensor;
    /// <summary>
    /// Base class from which all layers are derived.
    /// </summary>
    abstract class Layer:IModule
    {
        // class properties
        /// <summary>
        /// The size of each input sample
        /// </summary>
        public int InputSize { get; protected set; }
        /// <summary>
        /// The size of each output sample
        /// </summary>
        public int OutputSize { get; protected set; }

        /// <summary>
        /// Method to be called during forward-propagation.
        /// </summary>
        /// <param name="x">The input sample</param>
        /// <returns>Tensor of this layers output sample.</returns>
        public abstract Tensor Forward(Tensor x);

    }
}
