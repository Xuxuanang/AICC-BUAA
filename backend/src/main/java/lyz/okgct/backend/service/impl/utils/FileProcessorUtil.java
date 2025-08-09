package lyz.okgct.backend.service.impl.utils;

import org.springframework.stereotype.Component;

import java.io.*;

@Component
public class FileProcessorUtil {
    public static void WriteStringToFile(String code, String filePath) {
        try {
            FileWriter fileWriter = new FileWriter(filePath);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write(code);
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String ReadFileToString(String filePath) {
        String result = "";

        try {
            FileReader fileReader = new FileReader(filePath);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                result += line + "\n"; // 将每一行添加到结果字符串中
            }
            bufferedReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return result;
    }
}
