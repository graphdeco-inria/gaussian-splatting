/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr
 */



#include "core/graphics/Shader.hpp"
#include "core/graphics/RenderUtility.hpp"
#include "core/graphics/Window.hpp"

#define SIBR_WRITESHADER(src) "#version 420 core\n" #src


//#define RenderUtility::camStubDrawSize() 0.10f



namespace sibr
{


	static const std::vector<float>&	getCameraStubVertices(float camStubSize = 0.1f)
	{
		std::vector<float> _vBuffer(3*5);
		_vBuffer[3*0+0]= 1*camStubSize; _vBuffer[3*0+1]= 1*camStubSize; _vBuffer[3*0+2]=-3*camStubSize;
		_vBuffer[3*1+0]=-1*camStubSize; _vBuffer[3*1+1]= 1*camStubSize; _vBuffer[3*1+2]=-3*camStubSize;
		_vBuffer[3*2+0]=-1*camStubSize; _vBuffer[3*2+1]=-1*camStubSize; _vBuffer[3*2+2]=-3*camStubSize;
		_vBuffer[3*3+0]= 1*camStubSize; _vBuffer[3*3+1]=-1*camStubSize; _vBuffer[3*3+2]=-3*camStubSize;
		_vBuffer[3*4+0]= 0*camStubSize; _vBuffer[3*4+1]= 0*camStubSize; _vBuffer[3*4+2]= 0*camStubSize;
		return _vBuffer;
	}

	static const std::vector<uint>&		getCameraStubIndices( void )
	{
		static std::vector<uint>	_iBuffer;

		if (_iBuffer.empty())
		{
			_iBuffer.resize(3*6);

			_iBuffer[3*0+0]=0; _iBuffer[3*0+1]=1; _iBuffer[3*0+2]=4;
			_iBuffer[3*1+0]=1; _iBuffer[3*1+1]=2; _iBuffer[3*1+2]=4;
			_iBuffer[3*2+0]=2; _iBuffer[3*2+1]=4; _iBuffer[3*2+2]=3;
			_iBuffer[3*3+0]=0; _iBuffer[3*3+1]=4; _iBuffer[3*3+2]=3;
			_iBuffer[3*4+0]=0; _iBuffer[3*4+1]=1; _iBuffer[3*4+2]=3;
			_iBuffer[3*5+0]=1; _iBuffer[3*5+1]=2; _iBuffer[3*5+2]=3;
		}
		return _iBuffer;
	}


	void RenderUtility::sendVertsTexToGPU(GLuint vertTexVBO, GLfloat vert[], GLfloat tcoord[], int svert, int stcoord) {
		glBindBuffer(GL_ARRAY_BUFFER, vertTexVBO);
		glBufferData(GL_ARRAY_BUFFER, svert+stcoord, NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, svert, vert);
		glBufferSubData(GL_ARRAY_BUFFER, svert, stcoord, tcoord);
	}


	/*static*/ void		RenderUtility::renderScreenQuad( bool reverse, GLfloat tex_coor[] )
	{
		static GLfloat vert[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
		static GLfloat tcoord[8];
		static GLuint indexVBO, VAO, vertTexVBO;

		static bool firstTime = true;
		
		if(reverse)
		{
			GLfloat tmp[] = { 0,1,  0,0,  1,0,  1,1 };
			if( tex_coor )
				std::memcpy(tmp, tex_coor, sizeof tmp );

			std::memcpy(tcoord, tmp, sizeof tcoord);
			if( !firstTime )  // re-transfer to GPUs
				sendVertsTexToGPU(vertTexVBO, vert, tcoord, sizeof(vert), sizeof(tcoord));
		}
		else
		{
			GLfloat tmp[] = { 0,0,  1,0,  1,1,  0,1 };
			if( tex_coor )
				std::memcpy(tmp, tex_coor, sizeof tmp );	

			std::memcpy(tcoord, tmp, sizeof tcoord);
			if( !firstTime )  // re-transfer to GPU
				sendVertsTexToGPU(vertTexVBO, vert, tcoord, sizeof(vert), sizeof(tcoord));
		}

		static GLuint  ind[] = { 0,1,2,  0,2,3 };

		if( firstTime ) {
			firstTime = false;

			glGenVertexArrays(1, &VAO);
			glBindVertexArray(VAO);

			glGenBuffers(1, &indexVBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(ind), ind, GL_STATIC_DRAW);

			glGenBuffers(1, &vertTexVBO);
			sendVertsTexToGPU(vertTexVBO, vert, tcoord, sizeof(vert), sizeof(tcoord));
		}

		glBindVertexArray(VAO);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexVBO);
		glBindBuffer(GL_ARRAY_BUFFER, vertTexVBO);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)sizeof(vert)	);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0);

		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);
	}

	void		RenderUtility::renderScreenQuad()
	{
		static GLfloat Fvert[] = { -1,-1,0,  1,-1,0,  1,1,0,  -1,1,0 };
		static GLfloat Ftcoord[] = { 0, 0, 1, 0, 1, 1, 0, 1 };
		static GLuint  Find[] = { 0,1,2,  0,2,3 };
		static GLuint FindexVBO, FVAO, FvertTexVBO;
		static bool FfirstTime = true;
		static int lastContextId = -1;

		//std::cout << lastContextId << " " << Window::contextId << std::endl;
		if (lastContextId != Window::contextId || FfirstTime) {
			lastContextId = Window::contextId;
			FfirstTime = false;

			glGenVertexArrays(1, &FVAO);
			glBindVertexArray(FVAO);

			glGenBuffers(1, &FindexVBO);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, FindexVBO);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Find), Find, GL_STATIC_DRAW);

			glGenBuffers(1, &FvertTexVBO);
			sendVertsTexToGPU(FvertTexVBO, Fvert, Ftcoord, sizeof(Fvert), sizeof(Ftcoord));

			glBindBuffer(GL_ARRAY_BUFFER, FvertTexVBO);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, (void*)sizeof(Fvert));

			glBindVertexArray(0);
		}

		
		glBindVertexArray(FVAO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, FindexVBO);
		
		const GLboolean cullingWasEnabled = glIsEnabled(GL_CULL_FACE);
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, (void*)0);

		if (!cullingWasEnabled) {
			glDisable(GL_CULL_FACE);
		}

		glBindVertexArray(0);
	}


	/*static*/ Mesh		RenderUtility::createCameraStub(float camStubSize)
	{
		
		Mesh m;
		m.vertices( getCameraStubVertices(camStubSize) );
		m.triangles( getCameraStubIndices() );
		return m;
	}

	/*static*/ Mesh		RenderUtility::createScreenQuad( void )
	{
		Mesh::Vertices v;
		v.emplace_back(-1.0f, -1.0f, 0.0f);
		v.emplace_back( 1.0f, -1.0f, 0.0f);
		v.emplace_back( 1.0f,  1.0f, 0.0f);
		v.emplace_back(-1.0f,  1.0f, 0.0f);
		Mesh::UVs tc;
		tc.emplace_back(0.0f,0.0f);
		tc.emplace_back(1.0f,0.0f);
		tc.emplace_back(1.0f,1.0f);
		tc.emplace_back(0.0f,1.0f);
		Mesh::Triangles t;
		t.emplace_back(0);
		t.emplace_back(1);
		t.emplace_back(2);
		t.emplace_back(0);
		t.emplace_back(2);
		t.emplace_back(3);


		Mesh m;
		m.vertices(v);
		m.texCoords(tc);
		m.triangles(t);
		return m;
	}

	Mesh::Ptr RenderUtility::createAxisGizmo()
	{
		const float arrowShift = 0.2f;
		const float arrowSpread = 0.1f;

		Mesh::Vertices v = {
			// Axis X
			{-1,0,0}, {1,0,0},
			// Arrow X
			{1.0f - arrowShift, -arrowSpread, 0.0f},
			{1.0f - arrowShift, 0.0f, -arrowSpread},
			{1.0f - arrowShift, arrowSpread, 0.0f},
			{1.0f - arrowShift, 0.0f, arrowSpread},

			// Axis Y
			{0,-1,0}, {0,1,0},
			// Arrow Y
			{-arrowSpread, 1.0f - arrowShift, 0.0f},
			{0.0f, 1.0f - arrowShift, -arrowSpread},
			{arrowSpread, 1.0f - arrowShift, 0.0f},
			{0.0f, 1.0f - arrowShift, arrowSpread},

			// Axis Z
			{0, 0, -1}, {0, 0, 1},
			// Arrow Z
			{-arrowSpread, 0.0f, 1.0f - arrowShift},
			{0.0f, -arrowSpread, 1.0f - arrowShift},
			{arrowSpread, 0.0f, 1.0f - arrowShift},
			{0.0f, arrowSpread, 1.0f - arrowShift},

			// Letter X
			{1.0f + arrowShift - arrowSpread, -arrowSpread, 0.0f},
			{1.0f + arrowShift + arrowSpread, arrowSpread, 0.0f},
			{1.0f + arrowShift - arrowSpread, arrowSpread, 0.0f},
			{1.0f + arrowShift + arrowSpread, -arrowSpread, 0.0f},
			// Letter Y
			{0.0f, 1.0f + arrowShift - arrowSpread, 0.0f},
			{0.0f, 1.0f + arrowShift, 0.0f},
			{-arrowSpread, 1.0f + arrowShift + arrowSpread, 0.0f},
			{arrowSpread, 1.0f + arrowShift + arrowSpread, 0.0f},
			// Letter Z
			{0.0f, -arrowSpread, 1.0f + arrowShift - arrowSpread},
			{0.0f, -arrowSpread, 1.0f + arrowShift + arrowSpread},
			{0.0f, arrowSpread, 1.0f + arrowShift - arrowSpread},
			{0.0f, arrowSpread, 1.0f + arrowShift + arrowSpread}
		};

		Mesh::Colors c = {
			// Colors X
			{1, 0, 0}, {1, 0, 0}, {1, 0, 0},
			{1, 0, 0}, {1, 0, 0}, {1, 0, 0},
			// Colors Y
			{0, 1, 0}, {0, 1, 0}, {0, 1, 0},
			{0, 1, 0}, {0, 1, 0}, {0, 1, 0},
			// Colors Z
			{0, 0, 1}, {0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1}, {0, 0, 1},
			// Colors Letter X
			{1, 0, 0}, {1, 0, 0},
			{1, 0, 0}, {1, 0, 0},
			// Colors Letter Y
			{0, 1, 0}, {0, 1, 0},
			{0, 1, 0}, {0, 1, 0},
			// Colors Letter Z
			{0, 0, 1}, {0, 0, 1},
			{0, 0, 1}, {0, 0, 1}
		};

		Mesh::Triangles t = {
			// Axis X
			{0, 1, 0},
			// Arrow X
			{1, 2, 3}, {1, 3, 4}, {1, 4, 5},
			{1, 5, 2}, {2, 3, 4}, {2, 3, 5},
			// Axis Y
			{6, 7, 6},
			// Arrow Y
			{7, 8, 9}, {7, 9, 10}, {7, 10, 11},
			{7, 11, 8}, {8, 9, 10}, {8, 9, 11},
			// Axis Z
			{12, 13, 12},
			// Arrow Z
			{13, 14, 15}, {13, 15, 16}, {13, 16, 17},
			{13, 17, 14}, {14, 15, 16}, {14, 15, 17},

			// Letter X
			{18, 19, 18}, {20, 21, 20},
			//Letter Y
			{22, 23, 22}, {24, 23, 24}, {25, 23, 25},
			//Letter Z
			{26, 28, 26}, {26, 29, 26}, {27, 29, 27}
		};


		auto out = std::make_shared<sibr::Mesh>();
		out->vertices(v);
		out->colors(c);
		out->triangles(t);
		return out;
	}


	/*static*/ void		RenderUtility::useDefaultVAO( void )
	{
		static GLuint gDefaultVAO = 0;

		if (!gDefaultVAO)
			glGenVertexArrays(1, &gDefaultVAO);

		glBindVertexArray(gDefaultVAO);
	}


} // namespace sibr
