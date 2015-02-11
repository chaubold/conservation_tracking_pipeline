#include <iostream>
#include <vector>
#include <math.h>

// vigra
#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/accumulator.hxx>

// boost
#include <boost/shared_ptr.hpp>

// pgmlink
#include <pgmlink/feature.h>
#include <pgmlink/feature_calculator.h>
#include <pgmlink/feature_extraction.h>
#include <pgmlink/traxels.h>

#include "division_feature_extractor.hxx"

namespace fe = pgmlink::feature_extraction;

// code copied from https://github.com/martinsch/pgmlink-topsecret/blob/cvpr14/track_features/include/pgmlink/extract_region_features.h
// and adjusted to create traxels directly
namespace pgmlink {
namespace features{
	template<typename T>
	void set_feature(pgmlink::Traxel& t, const std::string& name, T value)
	{
		pgmlink::feature_array f;
		f.push_back(feature_type(value));
		t.features[name] = f;
	}

	template<>
	void set_feature(pgmlink::Traxel& t, const std::string &name, vigra::TinyVector<double,2> value)
	{
		pgmlink::feature_array f;
		f.clear();
		f.push_back(feature_type(value[0]));
		f.push_back(feature_type(value[1]));
		t.features[name] = f;
	}

	template<>
	void set_feature(pgmlink::Traxel& t, const std::string &name, vigra::TinyVector<double,3> value)
	{
		pgmlink::feature_array f;
		f.clear();
		f.push_back(feature_type(value[0]));
		f.push_back(feature_type(value[1]));
		f.push_back(feature_type(value[2]));
		t.features[name] = f;
	}

	template<int N, typename T1, typename T2>
	void set_feature_with_offset(pgmlink::Traxel& t,
		const std::string &name,
		vigra::TinyVector<T1,N> value,
		vigra::TinyVector<T2,N> offset)
	{
		pgmlink::feature_array f;
		f.clear();
		for(int i = 0; i < N; i++)
		{
			f.push_back(feature_type(value[i] + offset[i]));
		}
		t.features[name] = f;
	}

	template<>
	void set_feature(pgmlink::Traxel& t, const std::string &name, vigra::linalg::Matrix<double> value)
	{
		pgmlink::feature_array f;
		f.clear();
		for(auto it = value.begin(); it != value.end(); ++it)
		{
			f.push_back(*it);
		}
		t.features[name] = f;
	}

	///
	/// Extract features from the selected regions in the given MultiArrayView
	/// and insert them into the corresponding traxels (with offset) at the current timestep.
	/// Coordinates are shifted by the given offset if this is only a ROI of the full image.
	/// \return the maximal label id
	///
	template<int N, typename DataType, typename LabelType>
	void extract_region_features_roi(
		std::vector<pgmlink::Traxel>& traxels,
		const vigra::MultiArrayView<N, DataType>& data,
		const vigra::MultiArrayView<N, LabelType>& labels,
		const std::vector<size_t>& label_indices,
		unsigned int traxel_index_offset,
		const vigra::TinyVector<size_t, N>& coord_offsets,
		unsigned int timestep,
		bool verbose = true)
	{
		traxels.clear();

		// extract features using vigra
		using namespace vigra::acc;
		typedef AccumulatorChainArray<vigra::CoupledArrays<N, DataType, LabelType>,
			Select< // what statistics to compute (same as in joint seg & track, but without quantiles atm)
			RegionCenter,
			Count,
			Variance,
			Sum,
			Mean,
			RegionRadii,
			Central< PowerSum<2> >,
			Central< PowerSum<3> >,
			Central< PowerSum<4> >,
			Kurtosis,
			Maximum,
			Minimum,
			RegionAxes,
			Skewness,
			Weighted<PowerSum<0> >,
			Coord< Minimum >,
			Coord< Maximum >,
			DataArg<1>,
			LabelArg<2> // where to look for data and region labels
		> > FeatureAccumulator;
		FeatureAccumulator a;
		
		a.ignoreLabel(0); // do not compute features for the background
		LOG(pgmlink::logDEBUG1) << "Beginning feature extraction for frame " << timestep;
		extractFeatures<N, DataType, vigra::StridedArrayTag, LabelType, vigra::StridedArrayTag>(data, labels, a);
		LOG(pgmlink::logDEBUG1) << "Finished feature extraction for frame " << timestep;

		for(LabelType label : label_indices)
		{
			if(label == 0) // ignore background
				continue;
			// get respective feature map from FeatureStore
			//pgmlink::FeatureMap& feature_map = fs->get_traxel_features(timestep, label + traxel_index_offset);
			pgmlink::Traxel t;
			t.Id = label + traxel_index_offset;
			t.Timestep = timestep;

			// add features
			set_feature(t, "Mean", get<Mean>(a, label));
			set_feature(t, "Sum", get<Sum>(a, label));
			set_feature(t, "Variance", get<Variance>(a, label));
			set_feature(t, "Count", get<Count>(a, label));
			set_feature(t, "RegionRadii", get<RegionRadii>(a, label));
			set_feature_with_offset(t, "RegionCenter", get<RegionCenter>(a, label), coord_offsets);
			set_feature_with_offset(t, "Coord< Maximum >", get<Coord< Maximum > >(a, label), coord_offsets);
			set_feature_with_offset(t, "Coord< Minimum >", get<Coord< Minimum > >(a, label), coord_offsets);
			set_feature(t, "RegionAxes", get<RegionAxes>(a, label));
			set_feature(t, "Kurtosis", get<Kurtosis>(a, label));
			set_feature(t, "Minimum", get<Minimum>(a, label));
			set_feature(t, "Maximum", get<Maximum>(a, label));
			set_feature(t, "Skewness", get<Skewness>(a, label));
			set_feature(t, "Central< PowerSum<2> >", get<Central< PowerSum<2> > >(a, label));
			set_feature(t, "Central< PowerSum<3> >", get<Central< PowerSum<3> > >(a, label));
			set_feature(t, "Central< PowerSum<4> >", get<Central< PowerSum<4> > >(a, label));
			set_feature(t, "Weighted<PowerSum<0> >", get<Weighted<PowerSum<0> > >(a, label));
			traxels.push_back(t);
		}
	}
	
	///
	/// Extract features from all regions in the given MultiArrayView
	/// and insert them into the corresponding traxels at the current timestep.
	/// \return the maximal label id
	///
	template<int N, typename DataType, typename LabelType>
	int extract_region_features(
		std::vector<pgmlink::Traxel>& traxels,
		const vigra::MultiArrayView<N, DataType>& data,
		const vigra::MultiArrayView<N, LabelType>& labels,
		unsigned int timestep)
	{
		// create list of all labels we want to extract features for
		LabelType label_min, label_max;
		labels.minmax(&label_min, &label_max);
		std::vector<size_t> label_indices(label_max + 1 - label_min);
		std::iota(label_indices.begin(), label_indices.end(), label_min);
		// coordinate offset is zero, use default constructor
		vigra::TinyVector<size_t, N> coord_offset;
		extract_region_features_roi<N, DataType, LabelType>(traxels, data, labels, label_indices, 0, coord_offset, timestep, false);
		return label_max;
	}
} // namespace features
} // namespace pgmlink

template<int N, class DataType>
std::set<DataType> find_unique_labels_in_roi(vigra::MultiArrayView<N, DataType> roi, bool ignore_label_zero = true)
{
	typedef std::set<DataType> SetType;
	SetType labels(roi.begin(), roi.end());

	if(ignore_label_zero)
	{
		typename SetType::iterator it = std::find(labels.begin(), labels.end(), 0);
		if(it != labels.end())
			labels.erase(it);
	}

	return std::move(labels);
}

std::ostream& operator<<(std::ostream& lhs, const pgmlink::feature_array& rhs)
{
	if(rhs.size() == 0)
		return lhs;

	lhs << "(";
	for(float f : rhs)
	{
		lhs << f << ", ";
	}
	lhs << ")";
	return lhs;
}

int main(int argc, char** argv) {
	if(argc != 3)
	{
		std::cout << "Wrong number of arguments, 3 expected" << std::endl;
		return -1;
	}

	// mimics the division feature computation in ilastik/ilastik/applets/trackingFeatureExtraction/trackingFeatures.py
	// TODO: incorporate scale factor for anisotropic data

	// load raw image
	std::string image_filename(argv[1]);
	vigra::ImageImportInfo info(image_filename.c_str());
    vigra::Shape2 shape(info.width(), info.height());
	vigra::MultiArray<2, vigra::UInt8> image(shape);
	vigra::importImage(info, image);

	// load label image
	std::string label_image_filename(argv[2]);
	vigra::ImageImportInfo label_info(label_image_filename.c_str());
    vigra::Shape2 label_shape(label_info.width(), label_info.height());
	vigra::MultiArray<2, vigra::UInt8> label_image(label_shape);
	vigra::importImage(label_info, label_image);
	std::cout << "Image has shape: " << shape << std::endl;

	// compute region features
	std::vector<pgmlink::Traxel> traxels_f0;
	pgmlink::features::extract_region_features<2, vigra::UInt8, vigra::UInt8>(traxels_f0, image, label_image, 0);

	std::vector<pgmlink::Traxel> traxels_f1;
	pgmlink::features::extract_region_features<2, vigra::UInt8, vigra::UInt8>(traxels_f1, image, label_image, 1);

	// for each traxel, find candidates in next frame
	size_t template_size = 100;

	isbi_pipeline::DivisionFeatureExtractor<2, vigra::UInt8> dfe(template_size);
	dfe.extract(traxels_f0, traxels_f1, label_image);
}