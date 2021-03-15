namespace NEA.NeuralNetwork
{
    interface IModule
    {
        public Tensor.Tensor Forward(Tensor.Tensor input);
    }
}
