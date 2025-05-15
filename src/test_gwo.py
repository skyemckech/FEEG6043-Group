import g2o

def test_g2o():
    optimizer = g2o.SparseOptimizer()
    v1 = g2o.VertexSE2()
    v1.set_id(0)
    v1.set_estimate(g2o.SE2(0,0,0))
    optimizer.add_vertex(v1)
    
    v2 = g2o.VertexSE2()
    v2.set_id(1)
    v2.set_estimate(g2o.SE2(1,0,0))
    optimizer.add_vertex(v2)
    
    e = g2o.EdgeSE2()
    e.set_vertex(0, v1)
    e.set_vertex(1, v2)
    e.set_measurement(g2o.SE2(1,0,0))
    info = g2o.Matrix3d()
    info.setIdentity()
    e.set_information(info)  # Should NEVER crash
    optimizer.add_edge(e)
    print("Basic test passed")

test_g2o()