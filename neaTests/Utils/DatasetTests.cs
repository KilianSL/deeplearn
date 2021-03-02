using Microsoft.VisualStudio.TestTools.UnitTesting;
using NEA.Utils.Data;

namespace neaTests
{
    [TestClass]
    public class DatasetTests
    {
        // TestContext object for logging to the debug log within a test environment
        public TestContext TestContext { get; set; }

        // Test environment variables
        private readonly string path = @"C:\Users\kilian\source\Git Repos\WCGS-2021-6C-seifk\neaTests\testdataset.csv";

        private DataSet testDataset;

        [TestMethod]
        public void LoadData_ExpectedData()
        {
            testDataset = new DataSet();
            testDataset.LoadData(path, dataAnnotations: true);
            CollectionAssert.AreEqual(testDataset.ToArray(), new float?[][] { new float?[] { 1, 2, 3 }, new float?[] { 4, 5, 6 }, new float?[] { 7, null, 9 } });
        }

        [TestMethod]
        public void Clean_Expected()
        {
            testDataset.Clean();
            CollectionAssert.AreEqual(testDataset.ToArray(), new float?[][] { new float?[] { 1, 2, 3 }, new float?[] { 4, 5, 6 } });
        }

        [TestMethod]
        public void DeleteColumn_ExpectedData()
        {
            testDataset.RemoveFeature(2);
            CollectionAssert.AreEqual(testDataset.ToArray(), new float?[][] { new float?[] { 1, 2}, new float?[] { 4, 5} });
        }

        [TestMethod]
        public void RandomSample_ExpectedData() // Only really checks if the method is throwing an error
        {
            var sample = testDataset.RandomSample(1);
            CollectionAssert.Contains(testDataset.ToArray(), sample[0]);
        }
    }
}