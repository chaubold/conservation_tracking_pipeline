// stl
#include <iostream>

// own
#include "division_feature_extractor.hxx"

namespace isbi_pipeline 
{

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

} // namespace isbi_pipeline
