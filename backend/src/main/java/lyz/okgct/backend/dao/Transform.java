package lyz.okgct.backend.dao;

import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableId;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.io.Serializable;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Transform implements Serializable {
    @TableId(type = IdType.AUTO)
    private Integer id;

    private Integer userId;

    private String title;

    private String sourceFramework;

    private String sourceVersion;

    private String sourceCode;

    private String targetFramework;

    private String targetVersion;

    private String targetCode;

    private String codeFrom;

    private String models;

    private String sourceTree;

    private String targetTree;
}
