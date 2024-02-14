import { useBackend } from "../../../services/backend";
import { paths } from "../../../services/backend/endpoints";
import { BorderBox } from "../../atoms/BorderBox";
import { SectionHeader } from "../../atoms/SectionHeader";
import { Button } from "../../molecules/Button";
import { HorizontalRadio } from "../../molecules/HorizontalRadio";
import { InputInt } from "../../molecules/InputNumber";
import { ToggleSwitch } from "../../molecules/ToggleSwitch";
import { LabeledConfig } from "./LabeledConfig";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Props {
  image_id: number;
  initialSource: {
    width: number;
    height: number;
  };
  initialExtension: "JPEG" | "PNG" | "WEBP";
  onSuccessSubmit?: () => void;
}

export function ConfigurationsForm({
  image_id,
  initialSource,
  initialExtension,
  onSuccessSubmit,
}: Props) {
  const { backend } = useBackend();
  const queryClient = useQueryClient();
  const { mutate: runProcess, isPending } = useMutation({
    mutationFn: () =>
      backend
        .post<
          paths["/images/{id}/process"]["post"]["responses"]["200"]["content"]["application/json"]
        >(`/images/${image_id}/process`, configurations)
        .then((_) => _.data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: [`/images/${image_id}/process`],
      });
      onSuccessSubmit && onSuccessSubmit();
    },
  });

  const [configurations, setConfigurations] = useState<
    paths["/images/{id}/process"]["post"]["requestBody"]["content"]["application/json"]
  >({ extension: initialExtension, target: initialSource, enable_ai: false });
  const [preserveRatio, setPreserveRatio] = useState(true);

  const handleExtensionChange = (extension: "JPEG" | "PNG" | "WEBP") =>
    setConfigurations({ ...configurations, extension });
  const handleTargetWidthChange = (newWidth: number) => {
    let newHeight = preserveRatio
      ? Math.round(initialSource.height * (newWidth / initialSource.width))
      : configurations.target.height;
    newHeight = Math.min(9999, Math.max(1, newHeight));
    setConfigurations({
      ...configurations,
      target: { width: newWidth, height: newHeight },
    });
  };
  const handleTargetHeightChange = (newHeight: number) => {
    let newWidth = preserveRatio
      ? Math.round(initialSource.width * (newHeight / initialSource.height))
      : configurations.target.width;
    newWidth = Math.min(9999, Math.max(1, newWidth));
    setConfigurations({
      ...configurations,
      target: { width: newWidth, height: newHeight },
    });
  };
  const handleEnableAIChange = (value: boolean) =>
    setConfigurations({ ...configurations, enable_ai: value });

  return (
    <BorderBox className="w-72 space-y-3 bg-black p-4">
      <SectionHeader name="Configurations" />
      <LabeledConfig label="Source" disabled>
        <div className="flex items-center space-x-1">
          <InputInt value={initialSource.width} disabled className="w-12" />
          <span>x</span>
          <InputInt value={initialSource.height} disabled className="w-12" />
          <span>px</span>
        </div>
      </LabeledConfig>
      <LabeledConfig label="Extension">
        <HorizontalRadio
          possibleValues={["JPEG", "PNG", "WEBP"]}
          value={configurations.extension}
          setValue={handleExtensionChange}
          className="w-40"
        />
      </LabeledConfig>
      <LabeledConfig label="Preserve ratio">
        <ToggleSwitch checked={preserveRatio} onSwitch={setPreserveRatio} />
      </LabeledConfig>
      <LabeledConfig label="Target">
        <div className="flex items-center space-x-1">
          <InputInt
            value={configurations.target.width}
            min={1}
            max={9999}
            onChange={handleTargetWidthChange}
            className="w-12"
          />
          <span>x</span>
          <InputInt
            value={configurations.target.height}
            min={1}
            max={9999}
            onChange={handleTargetHeightChange}
            className="w-12"
          />
          <span>px</span>
        </div>
      </LabeledConfig>
      <LabeledConfig label="Enable AI (only on upscale)">
        <ToggleSwitch
          checked={configurations.enable_ai}
          onSwitch={handleEnableAIChange}
        />
      </LabeledConfig>
      <Button
        label="Let's run!"
        isLoading={isPending}
        onClick={() => runProcess()}
      />
    </BorderBox>
  );
}
