import { ProcessOptions } from "@/v2/features/processes/components/organisms/ProcessRadio/ProcessOptions";
import { HorizontalRadio } from "@/v2/features/shared/components/molecules/HorizontalRadio";

interface Props {
  value: ProcessOptions;
  setValue: (value: ProcessOptions) => void;
}

export function ProcessRadio({ value, setValue }: Props) {
  const values = Object.values(ProcessOptions);
  return (
    <HorizontalRadio
      possibleValues={values}
      value={value}
      setValue={setValue}
      className="w-72"
    />
  );
}
