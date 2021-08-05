




void disp_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sx, float *sy, float *sz,
		float lame_lambda, float mu);

void strain_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sxx, float *syy, float *szz, float *sxy, float *syz, float *szx,
		float lame_lambda, float mu);

void stress_point_source(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *sxx, float *syy, float *szz, float *sxy, float *syz, float *szx,
		float lame_lambda, float mu);

void c_point_source_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *s,
		float lame_lambda, float mu);

void c_point_source_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float x2, float y2, float z2, float *s,
		float lame_lambda, float mu);

void c_ps_strain(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float *x2, float *y2, float *z2, float *s, float lame_lambda, float mu, int pop_num, int thread_num);

void c_ps_stress(float x1, float y1, float z1, float strike1, float dip1, float moment_strike, float moment_dip, float moment_inflation,
		float moment_open, float *x2, float *y2, float *z2, float *s, float lame_lambda, float mu, int pop_num, int thread_num);
