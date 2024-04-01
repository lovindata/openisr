import { BorderBoxAtm } from "@/v2/features/shared/components/atoms/BorderBoxAtm";
import { SvgIconAtm } from "@/v2/features/shared/components/atoms/SvgIconAtm";
import { useBackend } from "@/v2/services/backend";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useDropzone } from "react-dropzone";

export function UploaderSectionLocal() {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: uploadImages } = useMutation({
    mutationFn: async (files: File[]) => {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) formData.append("files", files[i]);
      return backend
        .post("/command/v1/images/upload-local", formData)
        .then(() => {});
    },
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["/query/v1/app/cards"] }),
  });

  const { getRootProps, getInputProps } = useDropzone({
    accept: {
      "image/png": [".png"],
      "image/jpg": [".jpg"],
      "image/jpeg": [".jpeg"],
      "image/webp": [".webp"],
    },
    onDrop: (acceptedFiles) => uploadImages(acceptedFiles),
  });

  return (
    <BorderBoxAtm dashed className="h-16 w-32">
      <div {...getRootProps()} className="h-full">
        <input {...getInputProps()} />
        <div className="mx-auto flex h-full w-min items-center space-x-3">
          <SvgIconAtm type="folder" className="h-8 w-8" />
          <label>Local upload</label>
        </div>
      </div>
    </BorderBoxAtm>
  );
}
