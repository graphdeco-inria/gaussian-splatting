
namespace sibr
{
	void GLShader::authorize(void) const
	{
		if (!m_Shader) {
			std::string message = "sibr::GLShader " + m_Name + " used without having been initialized";
			throw std::runtime_error(message);
		}
	}

	void GLShader::begin( void )
	{
		CHECK_GL_ERROR;
		authorize();
		glUseProgram(m_Shader);
		m_Active = true;
		CHECK_GL_ERROR;
	}

	void GLShader::end( void )
	{
		glUseProgram(0);
		m_Active = false;
		CHECK_GL_ERROR;
	}

	void GLParameter::authorize(void) const
	{
		if (m_Shader == NULL) {
			std::string message = "GLParameter " + m_Name + " does not have a valid shader program";
			throw std::runtime_error(message);
		}
		if (m_Strict && m_Handle == -1) {
			std::string message = "GLParameter " + m_Name + " used without having been initialized";
			throw std::runtime_error(message);
		}
		if (!m_Shader->isActive()) {
			std::string message = "GLParameter " + m_Name + " used with shader is not active";
			throw std::runtime_error(message);
		}
	}

	void	GLParameter::set( float f )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform1f(m_Handle,f);
	}

	void	GLParameter::set( float a,float b )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform2f(m_Handle,a,b);
	}

	void	GLParameter::set( float a, float b, float c )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform3f(m_Handle,a,b,c);
	}

	void	GLParameter::set( float a, float b, float c, float d )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform4f(m_Handle,a,b,c,d);
	}

	void	GLParameter::set( const float *matrix )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniformMatrix4fv(m_Handle,1,GL_TRUE,matrix); // row major
	}

	void	GLParameter::set( GLuint tex )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform1i(m_Handle,tex);
	}

	void	GLParameter::set( int v )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform1i(m_Handle,v);
	}

	void	GLParameter::set(bool b)
	{
		set((int)b);
	}

	void	GLParameter::setArray( const float *pv, int size )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform1fv(m_Handle,size,pv);
	}
	
	void	GLParameter::setArray( const std::vector<int>& pv, int size )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform1iv(m_Handle,size,&pv[0]);
	}

	void	GLParameter::set(const Vector2i& v)
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform2i(m_Handle, v[0], v[1]);
	}

	void	GLParameter::set(const Vector3i& v)
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform3i(m_Handle, v[0], v[1], v[2]);
	}

	void	GLParameter::set(const Vector4i& v)
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform4i(m_Handle, v[0], v[1], v[2], v[3]);
	}

	void	GLParameter::set( const Vector2f& v )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform2f(m_Handle,v[0],v[1]);
	}

	void	GLParameter::set( const Vector3f& v )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform3f(m_Handle,v[0],v[1],v[2]);
	}

	void	GLParameter::set( const Vector4f& v )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniform4f(m_Handle,v[0],v[1],v[2],v[3]);
	}

	void	GLParameter::set( const Matrix4f& m )
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniformMatrix4fv(m_Handle,1,GL_FALSE,m.data()); // row major
	}

	void	GLParameter::setMatrixArray(const float* m, int num)
	{
		authorize();
		if (!m_Strict && m_Handle == -1) return;
		glUniformMatrix4fv(m_Handle, num, GL_FALSE, m); // row major
	}

} // namespace sibr
