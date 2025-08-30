#include "legged_wbc/Types.h"

#include <Eigen/Dense>
#include <pinocchio/multibody/data.hpp>

using namespace legged;

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 6, 6> computeFloatingBaseCentroidalMomentumMatrixInverse(const Eigen::Matrix<SCALAR_T, 6, 6>& Ab) {
  const SCALAR_T mass = Ab(0, 0);
  Eigen::Matrix<SCALAR_T, 3, 3> Ab_22_inv = Ab.template block<3, 3>(3, 3).inverse();
  Eigen::Matrix<SCALAR_T, 6, 6> Ab_inv = Eigen::Matrix<SCALAR_T, 6, 6>::Zero();
  Ab_inv << 1.0 / mass * Eigen::Matrix<SCALAR_T, 3, 3>::Identity(), -1.0 / mass * Ab.template block<3, 3>(0, 3) * Ab_22_inv,
      Eigen::Matrix<SCALAR_T, 3, 3>::Zero(), Ab_22_inv;
  return Ab_inv;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename Derived>
Eigen::Block<const Derived, 3, 1> getContactForces(const Eigen::MatrixBase<Derived>& input, size_t contactIndex,
                                                         const size_t& numThreeDofContacts,
                                                         const size_t& numSixDofContacts){
  assert(input.cols() == 1);
  assert(contactIndex < numThreeDofContacts + numSixDofContacts);
  const size_t contactForceIndex = 3 * contactIndex;
  const size_t contactWrenchIndex = 3 * numThreeDofContacts + 6 * (contactIndex - numThreeDofContacts);
  const size_t startRow = (contactIndex < numThreeDofContacts) ? contactForceIndex : contactWrenchIndex;
  return Eigen::Block<const Derived, 3, 1>(input.derived(), startRow, 0);
}

template <typename Derived>
Eigen::Block<const Derived, 3, 1> getContactTorques(const Eigen::MatrixBase<Derived>& input, size_t contactIndex,
                                                          const size_t& numThreeDofContacts,
                                                          const size_t& numSixDofContacts){
  assert(input.cols() == 1);
  assert(contactIndex < numThreeDofContacts + numSixDofContacts);
  assert(contactIndex >= numThreeDofContacts);
  const size_t startRow = 3 * numThreeDofContacts + 6 * (contactIndex - numThreeDofContacts) + 3;
  return Eigen::Block<const Derived, 3, 1>(input.derived(), startRow, 0);
}


/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
template <typename SCALAR_T>
Eigen::Matrix<SCALAR_T, 6, 1> getNormalizedCentroidalMomentumRate(const scalar_t& robotMass,
                                                                  Eigen::Vector3d com,
                                                                  std::vector<Eigen::Vector3d> contact3DofPoss,
                                                                  std::vector<Eigen::Vector3d> contact6DofPoss,
                                                                  const Eigen::Matrix<SCALAR_T, Eigen::Dynamic, 1>& input) {
  const Eigen::Matrix<SCALAR_T, 3, 1> gravityVector(SCALAR_T(0.0), SCALAR_T(0.0), SCALAR_T(-9.81));
  Eigen::Matrix<SCALAR_T, 6, 1> centroidalMomentumRate;
  centroidalMomentumRate << robotMass * gravityVector, Eigen::Matrix<SCALAR_T, 3, 1>::Zero();

  for (size_t i = 0; i < contact3DofPoss.size(); i++) {
    const auto contactForceInWorldFrame = getContactForces(input, i, contact3DofPoss.size(), contact6DofPoss.size());
    const auto positionComToContactPointInWorldFrame = (contact3DofPoss[i] - com);
    centroidalMomentumRate.template head<3>() += contactForceInWorldFrame;
    centroidalMomentumRate.template tail<3>().noalias() += positionComToContactPointInWorldFrame.cross(contactForceInWorldFrame);
  }  // end of i loop

  for (size_t i = contact3DofPoss.size(); i < contact3DofPoss.size() + contact6DofPoss.size(); i++) {
    const auto contactForceInWorldFrame = getContactForces(input, i, contact3DofPoss.size(), contact6DofPoss.size());
    const auto contactTorqueInWorldFrame = getContactTorques(input, i, contact3DofPoss.size(), contact6DofPoss.size());
    const auto positionComToContactPointInWorldFrame = (contact6DofPoss[i-contact3DofPoss.size()] - com);
    centroidalMomentumRate.template head<3>() += contactForceInWorldFrame;
    centroidalMomentumRate.template tail<3>().noalias() +=
        positionComToContactPointInWorldFrame.cross(contactForceInWorldFrame) + contactTorqueInWorldFrame;
  }  // end of i loop

  // normalize by the total mass
  centroidalMomentumRate /= robotMass;

  return centroidalMomentumRate;
}