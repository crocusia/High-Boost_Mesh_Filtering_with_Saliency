#include "Mesh.h"
#include <queue>

//Mesh¿« Saliency ø¨ªÍ (by.jhkim) 
void Mesh::computeSaliency(void)
{
	computeShapeOperator();
	computeMeanCurvature();
	computeIncidentMatrix();	
	NormalizedSaliency();
}


void Mesh::computeDirection() {

	for (auto v : _vertices) {
		Vec3<double> gradient(0.0, 0.0, 0.0);
		for (auto nv : v->_nbVertices) {
			Vec3<double> diff = nv->_pos - v->_pos;
			double sdiff = nv->_saliency - v->_saliency;
			diff.Normalize();
			gradient += diff * sdiff;
		}

		gradient.Normalize();
		v->_direction = gradient;
	}

	for (auto f : _faces) {
		Vec3<double> avgd(0.0, 0.0, 0.0);
		for (auto v : f->_vertices) {
			avgd += v->_direction;
		}
		avgd /= 3;
		f->_direction = avgd;
	}

	for (auto f : _faces) {
		f->reflectDirection();
	}
}

void Mesh::computeSaliencyDirection(double alpha, double lambda) {
	clock_t start, end;
	double duration;
	start = clock();

	for (auto f : _faces) {
		double avg = 0.0;
		for (auto v : f->_vertices) {
			avg += v->_saliency;
		}
		avg /= 3;
		f->_saliency = avg;
	}
	computeDirection();
	amplifySaliency(alpha, lambda);

	end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("computeSaliencyDirection : %f sec\n", duration);
}

void Mesh::NormalizedSaliency(void) {
	double maxS = 0.0;
	double minS = 0.5;
	for (auto v : _vertices) {
		if (maxS < v->_saliency) {
			maxS = v->_saliency;
		}
		if (minS > v->_saliency) {
			minS = v->_saliency;
		}
	}

	for (auto v : _vertices) {
		v->_saliency = (v->_saliency - minS) / (maxS - minS);
	}
}

void Mesh::amplifySaliency(double alpha, double lambda) {
	vector<double> sali;
	sali.clear();
	for (auto f : _faces) {
		sali.push_back(f->_saliency);
	}
	sort(sali.begin(), sali.end());
	int a = _faces.size() * alpha;
	_lambdaSaliency = sali[a];
	for (auto f : _faces) {
		if(f->_saliency >= _lambdaSaliency){
			f->_amplifySaliency = lambda * f->_saliency;
		}
		else {
			f->_amplifySaliency = f->_saliency;
		}
	}
}

void Mesh::computeSmoothNormal(int smoothK) {
	clock_t start, end;
	double duration;
	start = clock();

	vector<Vec3<double>> smooths;
	for (int i = 0; i < smoothK; i++) {
		for (auto f : _faces) {
			auto result = f->calcSmooth();
			result.Normalize();
			smooths.push_back(result);
		}
		for (auto f : _faces) {
			f->_smooth = smooths[f->_index];
		}
		smooths.clear();
	}

	end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("calcSmoothNormal : %f sec\n", duration);
}

void Mesh::computeSmoothwithSaliency(int smoothK) {
	clock_t start, end;
	double duration;
	start = clock();
	vector<Vec3<double>> smooths;
	for (int i = 0; i < smoothK; i++) {
		for (auto f : _faces) {
			auto result = f->calcSmoothwithSaliency(_lambdaSaliency);
			result.Normalize();
			smooths.push_back(result);
		}
		for (auto f : _faces) {
			f->_smooth = smooths[f->_index];
		}
		smooths.clear();
	}
	end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("calcSmoothNormalwithSaliency : %f sec\n", duration);
}

void Mesh::computeBoostNormal(double threshold) {
	clock_t start, end;
	double duration;
	start = clock();

	for (auto f : _faces) {
		f->calcBoost(threshold);
	}

	end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("calcBoost : %f sec\n", duration);
}

void Mesh::applyBilateralFilter(double sigma_dist, double sigma_value) {
	clock_t start, end;
	double duration;
	start = clock();

	vector<Vec3<double>> bi_results;
	for (auto f : _faces) {
		auto b_result = f->bilateralFilter(sigma_dist, sigma_value);
		bi_results.push_back(b_result);
	}
	for (auto f : _faces) {
		f->_boost = bi_results[f->_index];
	}
	bi_results.clear();

	end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("applyBilateralFilter : %f sec\n", duration);
}

void Mesh::computeGradient(int iterate, double learningrate) {
	clock_t start, end;
	double duration;
	start = clock();

	for (int i = 0; i < iterate; i++) {
		for (auto v : _vertices) {
			if (v->_isStop == false) {
				for (auto nf : v->_nbFaces) {
					auto resultR = nf->calcAreaRi(v);
					auto resultS = nf->calcAreaSi(v);
					auto test = resultR - resultS;
					v->_gradient += (resultR - resultS);
				}
				v->_gradient = v->_gradient * 2;
			}
		}
		for (auto v : _vertices) {
			if (v->_isStop == false) {
				if (v->_gradient.GetNorm() >= 0.001) {
					v->_pos = v->_pos - v->_gradient * 0.01;
					v->_gradient.Clear();
				}
				/*else {
					v->_isStop = true;
				}*/
			}
		}
	}

	end = clock();
	duration = (double)(end - start) / CLOCKS_PER_SEC;
	printf("computeGradient : %f sec\n", duration);
}
void Mesh::highBoostFilterwithSaliency_CPU(int smoothK, double alpha, double lambda, double threshold, int iterate, double learningrate) {
	printf("highBoostFilterwithSaliency_CPU start\n");
	computeSaliencyDirection(alpha, lambda);
	computeSmoothwithSaliency(smoothK);
	computeBoostNormal(threshold);
	computeGradient(iterate, learningrate);
	computeNormal();
	printf("highBoostFilterwithSaliency_CPU end\n");
}