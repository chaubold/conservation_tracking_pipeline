#ifndef ISBI_COMMON_H
#define ISBI_COMMON_H

// stl
#include <vector>
#include <string>
#include <map>
#include <utility>

// boost
#include <boost/filesystem.hpp>

// vigra
#include <vigra/random_forest.hxx>

// pgmlink
#include <pgmlink/traxels.h>
#include <pgmlink/event.h>

namespace isbi_pipeline {

// Type for the image data and calculation precision of the feature
// calculators
typedef float DataType;

// Type for labels
typedef short unsigned int LabelType;

// Type for reading the feature configuration
typedef std::vector<std::pair<std::string, DataType> > StringDataPairVectorType;
typedef std::map<std::string, std::string> StringStringMapType;

// boost types
typedef boost::filesystem::path PathType;
typedef boost::filesystem::directory_entry DirectoryEntryType;
typedef boost::filesystem::directory_iterator DirectoryIteratorType;

// vigra types
typedef vigra::RandomForest<LabelType> RandomForestType;
typedef std::vector<RandomForestType> RandomForestVectorType;

// pgmlink types
typedef pgmlink::feature_type FeatureType;
typedef pgmlink::FeatureMap FeatureMapType;
typedef std::vector<FeatureType> FeatureArrayType;
typedef pgmlink::TraxelStore TraxelStoreType;
typedef std::vector<pgmlink::Event> EventVectorType;
typedef std::vector<EventVectorType> EventVectorVectorType;

// Traxel index and index mapping types 
typedef std::pair<int, unsigned> TraxelIndexType;
typedef std::map<TraxelIndexType, LabelType> TraxelTrackIndexMapType;
typedef std::vector<TraxelIndexType> TraxelIndexVectorType;
typedef std::map<LabelType, TraxelIndexVectorType> TrackTraxelIndexMapType;


} // end of namespace isbi_pipeline


#endif // ISBI_COMMON_H
