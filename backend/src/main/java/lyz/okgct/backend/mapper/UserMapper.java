package lyz.okgct.backend.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import lyz.okgct.backend.dao.User;
import org.apache.ibatis.annotations.Mapper;

@Mapper
public interface UserMapper extends BaseMapper<User> {
}
