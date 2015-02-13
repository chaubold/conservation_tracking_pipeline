#ifndef ISBI_DIVISION_FEATURE_EXTRACTOR_HXX
#define ISBI_DIVISION_FEATURE_EXTRACTOR_HXX

#include <functional>
#include <memory>
#include <vector>
#include <algorithm>

// pgmlink
#include <pgmlink/feature.h>
#include <pgmlink/feature_calculator.h>
#include <pgmlink/feature_extraction.h>
#include <pgmlink/traxels.h>

// vigra
#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

// own
#include "common.h"

namespace isbi_pipeline
{

template<unsigned int N, typename T>
void copy_vector(
	const std::vector<T>& vector,
	vigra::MultiArrayView<N, T>& multi_array_view);

template<int N, class LabelType>
class DivisionFeatureExtractor
{
public:
	typedef std::pair<size_t , float> TraxelWithDistance;

public:
	DivisionFeatureExtractor(
		const std::vector<std::string>& feature_selection,
		const RandomForestVectorType& random_forests,
		size_t template_size = 50);

	void extract(std::vector<pgmlink::Traxel>& traxels_current_frame,
				 std::vector<pgmlink::Traxel>& traxels_next_frame,
				 vigra::MultiArrayView<N, LabelType> label_image_next_frame);

private:
	pgmlink::feature_array parent_children_ratio_operation(const pgmlink::feature_array& a,
														   const pgmlink::feature_array& b,
														   const pgmlink::feature_array& c);

	pgmlink::feature_array parent_children_angle_operation(const pgmlink::feature_array& a,
														   const pgmlink::feature_array& b,
														   const pgmlink::feature_array& c);

	std::set<LabelType> find_unique_labels_in_roi(vigra::MultiArrayView<N, LabelType> roi,
												 bool ignore_label_zero = true);

	void compute_traxel_division_features(pgmlink::Traxel& t,
											std::vector<TraxelWithDistance>& nearest_neighbors,
											std::vector<pgmlink::Traxel>& traxels_next_frame);

	std::vector< TraxelWithDistance > find_nearest_neighbors(const pgmlink::feature_array& traxel_position,
															 std::vector<pgmlink::Traxel>& traxels_next_frame,
															 vigra::MultiArrayView<N, LabelType> label_image_next_frame);

	void get_division_probability(pgmlink::Traxel& traxel);

private:
	const std::vector<std::string>& feature_selection_;
	const RandomForestVectorType& random_forests_;
	size_t template_size_;
	std::shared_ptr<pgmlink::feature_extraction::FeatureCalculator> squared_distance_calculator_;
	std::shared_ptr<pgmlink::feature_extraction::FeatureCalculator> children_ratio_calculator_;
	std::shared_ptr<pgmlink::feature_extraction::FeatureCalculator> parent_children_ratio_calculator_;
	std::shared_ptr<pgmlink::feature_extraction::FeatureCalculator> parent_children_angle_calculator_;
};

std::ostream& operator<<(std::ostream& lhs, const pgmlink::feature_array& rhs);


// ---------------------------------------------------------------------------------------------------------------

template<unsigned int N, typename T>
void copy_vector(
	const std::vector<T>& vector,
	vigra::MultiArrayView<N, T>& multi_array_view)
{
	typename std::vector<T>::const_iterator v_it = vector.begin();
	typename vigra::MultiArrayView<N, T>::iterator m_it = multi_array_view.begin();
	for (; v_it != vector.end(); v_it++, m_it++) {
		*m_it = *v_it;
	}
}


template<int N, class LabelType>
DivisionFeatureExtractor<N, LabelType>::DivisionFeatureExtractor(
		const std::vector<std::string>& feature_selection,
		const RandomForestVectorType& random_forests,
		size_t template_size):
	feature_selection_(feature_selection),
	random_forests_(random_forests),
	template_size_(template_size),
	squared_distance_calculator_(new pgmlink::feature_extraction::SquareRootSquaredDifferenceCalculator()),
	children_ratio_calculator_(new pgmlink::feature_extraction::RatioCalculator())
{
	parent_children_ratio_calculator_ = std::shared_ptr<pgmlink::feature_extraction::FeatureCalculator>(new pgmlink::feature_extraction::TripletOperationCalculator(
		std::bind(&DivisionFeatureExtractor<N, LabelType>::parent_children_ratio_operation,
				  this,
				  std::placeholders::_1,
				  std::placeholders::_2,
				  std::placeholders::_3),
		"ParentChildrenRatio"));
	parent_children_angle_calculator_ = std::shared_ptr<pgmlink::feature_extraction::FeatureCalculator>(new pgmlink::feature_extraction::TripletOperationCalculator(
		std::bind(&DivisionFeatureExtractor<N, LabelType>::parent_children_angle_operation,
				  this,
				  std::placeholders::_1,
				  std::placeholders::_2,
				  std::placeholders::_3),
		"ParentChildrenAngle"));
}


// ---------------------------------------------------------------------------------------------------------------
template<int N, class LabelType>
pgmlink::feature_array DivisionFeatureExtractor<N, LabelType>::parent_children_angle_operation(const pgmlink::feature_array& a,
							const pgmlink::feature_array& b,
							const pgmlink::feature_array& c)
{
	assert(a.size() == b.size() && a.size() == c.size() && a.size() >= N);
	pgmlink::feature_array ret(1, 0.);

	vigra::TinyVector<float, N> v1, v2;
	for(size_t i = 0; i < N; i++)
	{
		v1[i] = b[i] - a[i];
		v2[i] = c[i] - a[i];
	}

	float length_product = sqrt(vigra::squaredNorm(v1) * vigra::squaredNorm(v2));
	if(length_product == 0)
	{
		ret[0] = 0;
	}
	else
	{
		ret[0] = acos(vigra::dot(v1, v2) / length_product) * 180.0 / M_PI;
	}

	return std::move(ret);
}


// ---------------------------------------------------------------------------------------------------------------
template<int N, class LabelType>
std::set<LabelType> DivisionFeatureExtractor<N, LabelType>::find_unique_labels_in_roi(vigra::MultiArrayView<N, LabelType> roi,
																		bool ignore_label_zero)
{
	typedef std::set<LabelType> SetType;
	SetType labels(roi.begin(), roi.end());

	if(ignore_label_zero)
	{
		typename SetType::iterator it = std::find(labels.begin(), labels.end(), 0);
		if(it != labels.end())
			labels.erase(it);
	}

	return std::move(labels);
}


// ---------------------------------------------------------------------------------------------------------------
template<int N, class LabelType>
void DivisionFeatureExtractor<N, LabelType>::extract(std::vector<pgmlink::Traxel>& traxels_current_frame,
										 std::vector<pgmlink::Traxel>& traxels_next_frame,
										 vigra::MultiArrayView<N, LabelType> label_image_next_frame)
{
	for(pgmlink::Traxel& t : traxels_current_frame)
	{
		std::cout << "Investigating " << t << " at com " << t.features["RegionCenter"] << std::endl;
		std::vector<TraxelWithDistance> nearest_neighbors = find_nearest_neighbors(t.features["RegionCenter"],
																				   traxels_next_frame,
																				   label_image_next_frame);

		std::cout << "Distances after sorting:" << std::endl;
		for(TraxelWithDistance& twd : nearest_neighbors)
		{
			std::cout << "\t" << traxels_next_frame[twd.first] << " at distance " << twd.second << std::endl;
		}

		compute_traxel_division_features(t, nearest_neighbors, traxels_next_frame);

	}
}


// ---------------------------------------------------------------------------------------------------------------
template<int N, class LabelType>
std::vector< typename DivisionFeatureExtractor<N, LabelType>::TraxelWithDistance >
DivisionFeatureExtractor<N, LabelType>::find_nearest_neighbors(
	const pgmlink::feature_array& traxel_position,
	std::vector<pgmlink::Traxel>& traxels_next_frame,
	vigra::MultiArrayView<N, LabelType> label_image_next_frame)
{
	// get ROI from next frame
	vigra::TinyVector<size_t, N> start, stop;

	for(int i = 0; i < N; i++)
	{
		start[i] = std::max(0, int(traxel_position[i] - template_size_ / 2));
		stop[i]  = std::min(int(label_image_next_frame.shape(0)), int(traxel_position[i] + template_size_ / 2));
	}

	vigra::MultiArrayView<N, LabelType> roi = label_image_next_frame.subarray(start, stop);

	// find all labels in this roi, and order them according to their distances
	std::set<LabelType> labels_in_roi = find_unique_labels_in_roi(roi);
	std::vector< TraxelWithDistance > nearest_neighbors;

	// compute distance for each of those
	for(LabelType label : labels_in_roi)
	{
		// find traxel with this label
		for(size_t trax_idx = 0; trax_idx < traxels_next_frame.size(); trax_idx++)
		{
			pgmlink::Traxel& tr = traxels_next_frame[trax_idx];
			if(tr.Id == label)
			{
				// compute distance
				pgmlink::feature_array distance = squared_distance_calculator_->calculate(traxel_position, tr.features["RegionCenter"]);
				// std::cout << "\tFound " << tr << " at distance " << distance[0] << std::endl;
				nearest_neighbors.push_back(std::make_pair(trax_idx, distance[0]));
				break;
			}
		}
	}

	// sort neighbors according to distance
	auto compareDistances = [](const TraxelWithDistance& a, const TraxelWithDistance& b){ return a.second < b.second; };
	std::sort(nearest_neighbors.begin(), nearest_neighbors.end(), compareDistances);

	return std::move(nearest_neighbors);
}


// ---------------------------------------------------------------------------------------------------------------
template<int N, class LabelType>
pgmlink::feature_array DivisionFeatureExtractor<N, LabelType>::parent_children_ratio_operation(const pgmlink::feature_array& a,
							const pgmlink::feature_array& b,
							const pgmlink::feature_array& c)
{
	assert(a.size() == b.size() && a.size() == c.size());
	pgmlink::feature_array ret(a.size(), 0.);
	for(size_t i = 0; i < a.size(); i++)
	{
		ret[i] = b[i] + c[i];
		if(ret[i] < 0.000001)
			ret[i] = 9999.0;
		else
			ret[i] = a[i] / ret[i];
	}
	return std::move(ret);
}


// ---------------------------------------------------------------------------------------------------------------
template<int N, class LabelType>
void DivisionFeatureExtractor<N, LabelType>::compute_traxel_division_features(pgmlink::Traxel& t,
	std::vector<TraxelWithDistance>& nearest_neighbors,
	std::vector<pgmlink::Traxel>& traxels_next_frame)
{
	// store squared distances in traxels
	for(size_t i = 0; i < nearest_neighbors.size(); i++)
	{
		std::stringstream feat_name;
		feat_name << "SquaredDistances_" << i;
		t.features[feat_name.str()] = { nearest_neighbors[i].second };
	}

	if(nearest_neighbors.size() > 1)
	{
		// compute remaining features:
		t.features["ChildrenRatio_Count"] = children_ratio_calculator_->calculate(
														traxels_next_frame[nearest_neighbors[0].first].features["Count"],
														traxels_next_frame[nearest_neighbors[1].first].features["Count"]);
		t.features["ChildrenRatio_Mean"] = children_ratio_calculator_->calculate(
														traxels_next_frame[nearest_neighbors[0].first].features["Mean"],
														traxels_next_frame[nearest_neighbors[1].first].features["Mean"]);
		t.features["ChildrenRatio_SquaredDistances"] = children_ratio_calculator_->calculate(
														t.features["SquaredDistances_0"],
														t.features["SquaredDistances_1"]);
		t.features["ParentChildrenRatio_Mean"] = parent_children_ratio_calculator_->calculate(
														t.features["Mean"],
														traxels_next_frame[nearest_neighbors[0].first].features["Mean"],
														traxels_next_frame[nearest_neighbors[1].first].features["Mean"]);
		t.features["ParentChildrenRatio_Count"] = parent_children_ratio_calculator_->calculate(
														t.features["Count"],
														traxels_next_frame[nearest_neighbors[0].first].features["Count"],
														traxels_next_frame[nearest_neighbors[1].first].features["Count"]);
		float angle = parent_children_angle_calculator_->calculate(
														t.features["RegionCenter"],
														traxels_next_frame[nearest_neighbors[0].first].features["RegionCenter"],
														traxels_next_frame[nearest_neighbors[1].first].features["RegionCenter"])[0];
		if(nearest_neighbors.size() > 2)
		{
			angle = std::max(angle, parent_children_angle_calculator_->calculate(
														t.features["RegionCenter"],
														traxels_next_frame[nearest_neighbors[1].first].features["RegionCenter"],
														traxels_next_frame[nearest_neighbors[2].first].features["RegionCenter"])[0]);
			angle = std::max(angle, parent_children_angle_calculator_->calculate(
														t.features["RegionCenter"],
														traxels_next_frame[nearest_neighbors[0].first].features["RegionCenter"],
														traxels_next_frame[nearest_neighbors[2].first].features["RegionCenter"])[0]);
		}
		t.features["ParentChildrenAngle_RegionCenter"] = { angle };
		get_division_probability(t);
	}
	else
	{
		t.features["ChildrenRatio_Count"] = {0.0f};
		t.features["ChildrenRatio_Mean"] = {0.0f};
		t.features["ChildrenRatio_SquaredDistances"] = {0.0f};
		t.features["ParentChildrenRatio_Mean"] = {0.0f};
		t.features["ParentChildrenRatio_Count"] = {0.0f};
		t.features["ParentChildrenAngle_RegionCenter"] = {0.0f};
		t.features["divProb"] = {0.0f};
	}
	std::cout << "\tChildrenRatio_Count" << t.features["ChildrenRatio_Count"] << std::endl;
	std::cout << "\tChildrenRatio_Mean" << t.features["ChildrenRatio_Mean"] << std::endl;
	std::cout << "\tParentChildrenRatio_Mean" << t.features["ParentChildrenRatio_Mean"] << std::endl;
	std::cout << "\tParentChildrenRatio_Count" << t.features["ParentChildrenRatio_Count"] << std::endl;
	std::cout << "\tParentChildrenAngle_RegionCenter" << t.features["ParentChildrenAngle_RegionCenter"] << std::endl;
	std::cout << "\tdivProb" << t.features["divProb"] << std::endl;
}

template<int N, class LabelType>
void DivisionFeatureExtractor<N, LabelType>::get_division_probability(
	pgmlink::Traxel& traxel)
{
	// get the size of the feature vector
	size_t feature_size = feature_selection_.size();
	// get all features into one multi array
	vigra::MultiArray<2, FeatureType> features(vigra::Shape2(1, feature_size));
	for(size_t offset = 0; offset < feature_size; offset++) {
		std::cout << "Fetch feature " << feature_selection_[offset] << std::endl;
		const std::string& feature_name = feature_selection_[offset];
		if (traxel.features.count(feature_name) == 0) {
			throw std::runtime_error("Feature " + feature_name + " not found");
			// TODO throw runtime error
			features(0, offset) = 0.0f;
		} else {
			features(0, offset) = traxel.features[feature_name][0];
		}
	}
	// evaluate the random forests
	vigra::MultiArray<2, FeatureType> probabilities(vigra::Shape2(1, 2), 0.0);
	for (size_t n = 0; n < random_forests_.size(); n++) {
		vigra::MultiArray<2, FeatureType> probabilities_temp(vigra::Shape2(1, 2));
		random_forests_[n].predictProbabilities(features, probabilities_temp);
		probabilities += probabilities_temp;
	}
	// fill the features map
	traxel.features["divProb"].clear();
	traxel.features["divProb"].push_back(probabilities(0, 1));
}

} // namespace isbi_pipeline

#endif // ISBI_DIVISION_FEATURE_EXTRACTOR_HXX
