using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NEA.Utils.Data
{
    /// <summary>
    /// A class representing a dataset, as well as allowing basic data interactions.
    /// </summary>
    public class DataSet
    {
        /// <summary>
        /// The number of entries in the dataset
        /// </summary>
        public int Count { get { return data.Length; } private set { } }

        /// <summary>
        /// The number of features in the dataset
        /// </summary>
        public int Features { get { return data[0].Length; } private set { } }

        // The data held by the dataset, as a nested array. Each sub-array is one row of data.
        // Nullable as the class uses the null value for data cleaning.
        private float?[][] data;

        // Constructors

        /// <summary>
        /// Creates a new empty instance of the dataset class
        /// </summary>
        public DataSet()
        {
            data = new float?[0][];
        }

        /// <summary>
        /// Creates a new instance of the dataset, populating it with data from the specified path.
        /// </summary>
        /// <param name="path">The path of the target dataset. Should specify a *.csv file.</param>
        /// <param name="dataAnnotations">Whether the target file has data annotations in the first line. Default false.</param>
        /// <param name="delimiter">The character used to separate fields on each row of the file. Default comma.</param>
        public DataSet(string path, bool dataAnnotations = false, char delimiter = ',')
        {
            // Calls loadData function to assign class attributes
            LoadData(path, dataAnnotations, delimiter);
        }

        /// <summary>
        /// Creates a dataset from a pre-populated nullable float array
        /// </summary>
        /// <param name="data">The data to populate the dataset with</param>
        public DataSet(float?[][] data)
        {
            this.data = data;
        }

        // Indexers
        /// <summary>
        /// Specifies a value at the specified row and column index.
        /// </summary>
        /// <param name="row">The row index.</param>
        /// <param name="column">The column index.</param>
        /// <returns>The nullable float at the specified index.</returns>
        public float? this[int row, int column]
        {
            get => data[row][column];
            set => data[row][column] = value;
        }

        /// <summary>
        /// Specifies a row at the specified index.
        /// </summary>
        /// <param name="index">The row index.</param>
        /// <returns>The array of nullable floats at the specified index.</returns>
        public float?[] this[int index]
        {
            get => data[index];
            set => data[index] = value;
        }

        /// <summary>
        /// Loads data from the file at the specified path.
        /// </summary>
        /// <param name="path">The path of the target dataset. Should specify a *.csv file.</param>
        /// <param name="dataAnnotations">Whether the target file has data annotations in the first line. Default false.</param>
        /// <param name="delimiter">The character used to separate fields on each row of the file. Default comma.</param>
        public void LoadData(string path, bool dataAnnotations = false, char delimiter = ',')
        {
            // Nested function that takes a split array of string data and converts it to an array of nullable floats
            // Added for DRY/readability
            static float?[] convertLineToDatarow(string[] line)
            {
                var dataRow = new float?[line.Length];
                for (int i = 0; i < line.Length; i++)
                {
                    // Uses tryparse to check if the value is valid, and assigns a value of null if it isn't
                    if (float.TryParse(line[i], out float tempFloatValue))
                    {
                        dataRow[i] = tempFloatValue;
                    }
                    else
                    {
                        dataRow[i] = null;
                    }
                }
                return dataRow;
            }

            using var reader = new StreamReader(path);

            // create buffer for first line, as reading the line from streamreader consumes it.
            // Reading the first line separately is necessary in the case that the dataset has annotations
            string firstLineBuffer = reader.ReadLine();
            string[] firstLineItems = firstLineBuffer.Split(delimiter);

            // Dynamic list used to accumulate the values in the dataset, converted to jagged array later
            // Each list entry is an array of nullable floats - this will become important when it comes to cleaning the data
            var datasetList = new List<float?[]>();

            // Adds the first line to the dataset if it is not data annotations.
            if (!dataAnnotations)
            {
                datasetList.Add(convertLineToDatarow(firstLineItems));
            }

            // Reads the rest of the file, converting it in the same manner
            while (!reader.EndOfStream)
            {
                var items = reader.ReadLine().Split(delimiter); // one-liner to read and split a line
                datasetList.Add(convertLineToDatarow(items));
            }

            // Close streamReader to deallocate memory
            reader.Close();

            // assigns class variables accordingly
            data = datasetList.ToArray();
        }

        /// <summary>
        /// Removes all elements with invalid entries from the dataset. Return the amount of items removed.
        /// </summary>
        public int Clean()
        {
            // Creates dynamic list from dataset -> easier to work with than an array
            var dataSetList = new List<float?[]>(data);
            // Removes any items from the list where the item array contains a null value
            int itemsRemoved = dataSetList.RemoveAll(entry => Array.Exists(entry, item => item == null));

            // Updated class attributes
            data = dataSetList.ToArray();

            return itemsRemoved;
        }

        /// <summary>
        /// Performs a Fisher-Yates shuffle on the dataset
        /// </summary>
        public void Shuffle()
        {
            // Initialises new random object
            var rand = new Random();

            // Shuffles the array of indexes
            for (int i = Count - 1; i >= 0; i--)
            {
                int swapIdx = rand.Next(0, i + 1);
                float?[] _ = data[i]; // temporary value
                data[i] = data[swapIdx];
                data[swapIdx] = _;
            }
        }

        /// <summary>
        /// Selects a specified number of items from the dataset.
        /// </summary>
        /// <param name="nItems">The number of items to select.</param>
        /// <returns>A random sample of items from the dataset.</returns>
        public float?[][] RandomSample(int nItems)
        {
            // Validation of nItems
            if (nItems > Count)
            {
                throw new Exception("Number of items in sample cannot be larger than dataset");
            }
            var sample = new float?[nItems][];
            var rand = new Random();
            var selectedItems = new List<int>();
            for (int i = 0; i < nItems; i++)
            {
                int index = rand.Next(0, Count - 1);
                while (selectedItems.Contains(index))
                {
                    index = rand.Next(0, Count - 1);
                }
                // Creates shallow copy for sample array to avoid modifying referenced data accidentally
                sample[i] = (float?[])data[index].Clone();
            }
            return sample;
        }

        /// <summary>
        /// Removes the feature at the specified column index from every element in the dataset
        /// </summary>
        /// <param name="index">The column index of the feature to remove</param>
        public void RemoveFeature(int index)
        {
            for (int i = 0; i < Count; i++)
            {
                // Creates a list from the element at that index. This allows the data to be used dynamically/.
                var item = new List<float?>(data[i]);
                item.RemoveAt(index);
                data[i] = item.ToArray();
            }
        }

        /// <summary>
        /// Removes a single item from the dataset at the specified index.
        /// </summary>
        /// <param name="index">The row index of the item to be removed.</param>
        public void RemoveElementAt(int index)
        {
            // Creates list of the data array, for easy element manipulation.
            var dataList = new List<float?[]>(data);
            dataList.RemoveAt(index);
            data = dataList.ToArray();
        }

        /// <summary>
        /// Splits the dataset into a training dataset and a test dataset.
        /// </summary>
        /// <param name="trainTestSplit">The proportion of the dataset which should be test data</param>
        /// <returns>A tuple of (trainingdata, testdata)</returns>
        public (DataSet train, DataSet test) TrainTestSplit(float trainTestSplit)
        {
            int splitPnt = (int)MathF.Floor(Count * trainTestSplit); // Finds index to split at
            // Uses linq to split the dataset around the split point
            var trainSet = data.Take(splitPnt).ToArray();
            var testSet = data.Skip(splitPnt).ToArray();
            return (new DataSet(trainSet), new DataSet(testSet));
        }

        /// <summary>
        /// Returns the contents of the dataset as a nullable float array.
        /// </summary>
        /// <returns></returns>
        public float?[][] ToArray()
        {
            return data;
        }
    }
}